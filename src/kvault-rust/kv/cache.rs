//! KV overlays. The writer connection serializes batch publication and retirement.

use std::collections::BTreeMap;
use std::ops::Bound::{Excluded, Included, Unbounded};
use std::sync::Arc;

use crate::common::CacheError;

pub(crate) type SharedValue = Arc<Vec<u8>>;
pub(crate) type CacheMap = BTreeMap<Vec<u8>, SharedValue>;

pub(crate) struct FlushBatch {
    pub(crate) map: CacheMap,
    bytes: usize,
}

pub(crate) struct KvCache {
    // Kept ordered so each scan copies only its requested page of references.
    pub(crate) map: CacheMap,
    pending_bytes: usize,
    inflight: Option<Arc<FlushBatch>>,
    cap_bytes: usize,
    flush_threshold: usize,
}

impl KvCache {
    pub(crate) fn new(cap_bytes: usize, flush_threshold: usize) -> Self {
        Self {
            map: BTreeMap::new(),
            pending_bytes: 0,
            inflight: None,
            cap_bytes,
            flush_threshold,
        }
    }

    pub(crate) fn stats(&self) -> (usize, usize, usize) {
        (
            self.map.len(),
            self.inflight.as_ref().map_or(0, |batch| batch.map.len()),
            self.pending_bytes + self.inflight.as_ref().map_or(0, |batch| batch.bytes),
        )
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.map.is_empty() && self.inflight.is_none()
    }

    pub(crate) fn should_flush(&self) -> bool {
        self.pending_bytes >= self.flush_threshold
    }

    pub(crate) fn insert(&mut self, key: &[u8], value: &SharedValue) -> Result<(), CacheError> {
        let size = key.len().saturating_add(value.len());
        if size > self.cap_bytes {
            return Err(CacheError::ValueTooLarge);
        }
        let replaced = self.map.get(key).map_or(0, |old| key.len() + old.len());
        let projected = (self.stats().2 - replaced).saturating_add(size);
        if projected > self.cap_bytes {
            return Err(CacheError::NeedFlush);
        }
        self.pending_bytes = self.pending_bytes - replaced + size;
        self.map.insert(key.to_vec(), Arc::clone(value));
        Ok(())
    }

    pub(crate) fn get(&self, key: &[u8]) -> Option<SharedValue> {
        self.map
            .get(key)
            .or_else(|| self.inflight.as_ref().and_then(|batch| batch.map.get(key)))
            .cloned()
    }

    pub(crate) fn contains_key(&self, key: &[u8]) -> bool {
        self.map.contains_key(key)
            || self
                .inflight
                .as_ref()
                .is_some_and(|batch| batch.map.contains_key(key))
    }

    // Direct mutations call this after commit while still holding the writer connection.
    pub(crate) fn remove(&mut self, key: &[u8]) -> Option<SharedValue> {
        debug_assert!(self.inflight.is_none());
        let value = self.map.remove(key)?;
        self.pending_bytes -= key.len() + value.len();
        Some(value)
    }

    pub(crate) fn take_pending(&mut self) -> CacheMap {
        debug_assert!(self.inflight.is_none());
        self.pending_bytes = 0;
        std::mem::take(&mut self.map)
    }

    pub(crate) fn keys(&self) -> Vec<Vec<u8>> {
        let pending = self.map.keys();
        let inflight = self.inflight.iter().flat_map(|batch| batch.map.keys());
        pending
            .chain(inflight)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .cloned()
            .collect()
    }

    pub(crate) fn page(
        &self,
        prefix: Option<&[u8]>,
        after: Option<&[u8]>,
        limit: Option<usize>,
        include_values: bool,
    ) -> CacheMap {
        let lower = match (prefix, after) {
            (Some(prefix), Some(after)) if after < prefix => Included(prefix),
            (_, Some(after)) => Excluded(after),
            (Some(prefix), None) => Included(prefix),
            (None, None) => Unbounded,
        };
        let mut page = BTreeMap::new();
        let empty = Arc::new(Vec::new());
        // Each source contributes at most N entries; pending overwrites in-flight.
        for map in self
            .inflight
            .iter()
            .map(|batch| &batch.map)
            .chain(std::iter::once(&self.map))
        {
            for (key, value) in map
                .range::<[u8], _>((lower, Unbounded))
                .take_while(|(key, _)| prefix.is_none_or(|prefix| key.starts_with(prefix)))
                .take(limit.unwrap_or(usize::MAX))
            {
                page.insert(key.clone(), Arc::clone(if include_values { value } else { &empty }));
            }
        }
        // Limit the overlay itself before merging with the committed page.
        if let Some(limit) = limit {
            while page.len() > limit {
                page.pop_last();
            }
        }
        page
    }

    // The caller must hold the writer connection until finish_flush completes.
    pub(crate) fn begin_flush(&mut self) -> Option<Arc<FlushBatch>> {
        debug_assert!(self.inflight.is_none());
        if self.map.is_empty() {
            return None;
        }
        let batch = Arc::new(FlushBatch {
            map: std::mem::take(&mut self.map),
            bytes: std::mem::take(&mut self.pending_bytes),
        });
        self.inflight = Some(Arc::clone(&batch));
        Some(batch)
    }

    // Return retired storage so its destruction happens outside the cache lock.
    pub(crate) fn finish_flush(&mut self, committed: bool) -> Option<Arc<FlushBatch>> {
        let batch = self.inflight.take()?;
        if committed {
            return Some(batch);
        }
        // Only the flush owns the batch, and drops its reference before recovery.
        let batch =
            Arc::try_unwrap(batch).unwrap_or_else(|_| unreachable!("flush batch still borrowed"));
        for (key, value) in batch.map {
            if let std::collections::btree_map::Entry::Vacant(entry) = self.map.entry(key) {
                self.pending_bytes += entry.key().len() + value.len();
                entry.insert(value);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_insert_and_limits() {
        let mut cache = KvCache::new(100, 50);
        assert!(cache.is_empty());
        assert!(!cache.should_flush());
        cache.insert(b"k1", &Arc::new(vec![0; 40])).unwrap();
        assert_eq!(cache.stats(), (1, 0, 42));
        assert!(!cache.is_empty());
        assert!(matches!(
            cache.insert(b"large", &Arc::new(vec![0; 200])),
            Err(CacheError::ValueTooLarge)
        ));
        assert!(matches!(
            cache.insert(b"k2", &Arc::new(vec![0; 70])),
            Err(CacheError::NeedFlush)
        ));
        cache.insert(b"k1", &Arc::new(vec![1; 50])).unwrap();
        assert!(cache.should_flush());
        assert_eq!(cache.stats(), (1, 0, 52));
    }

    #[test]
    fn committed_batch_retires_without_losing_pending() {
        let mut cache = KvCache::new(1000, 500);
        cache
            .insert(b"key1", &Arc::new(b"value1".to_vec()))
            .unwrap();
        cache
            .insert(b"key2", &Arc::new(b"value2".to_vec()))
            .unwrap();
        let batch = cache.begin_flush().unwrap();
        assert_eq!(batch.map.len(), 2);
        assert_eq!(cache.stats(), (0, 2, 20));
        assert_eq!(cache.get(b"key1").unwrap().as_slice(), b"value1");
        drop(batch);
        drop(cache.finish_flush(true));
        assert!(cache.is_empty());
        assert_eq!(cache.stats(), (0, 0, 0));
    }

    #[test]
    fn failed_batch_preserves_newest_and_counts_inflight() {
        let mut cache = KvCache::new(100, 100);
        cache.insert(b"k", &Arc::new(vec![0; 40])).unwrap();
        drop(cache.begin_flush());
        cache.insert(b"k", &Arc::new(vec![1; 40])).unwrap();
        assert_eq!(cache.stats(), (1, 1, 82));
        assert!(matches!(
            cache.insert(b"other", &Arc::new(vec![0; 20])),
            Err(CacheError::NeedFlush)
        ));
        cache.finish_flush(false);
        assert_eq!(cache.stats(), (1, 0, 41));
        assert_eq!(cache.get(b"k").unwrap().as_slice(), &[1; 40]);
    }
}
