use parking_lot::Mutex;

pub struct Pool<T> {
    objects: Mutex<Vec<T>>,
}

impl<T: Default> Pool<T> {
    #[inline]
    pub fn new() -> Pool<T> {
        Pool {
            objects: Mutex::new(Vec::new()),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.objects.lock().len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.objects.lock().is_empty()
    }

    #[inline]
    pub fn get_object(&self) -> T {
        self.objects.lock().pop().unwrap_or_else(|| T::default())
    }

    #[inline]
    pub fn return_object(&self, t: T) {
        self.objects.lock().push(t)
    }
}

impl<T: Default> Default for Pool<T> {
    #[inline]
    fn default() -> Pool<T> {
        Pool::new()
    }
}

unsafe impl<T: Default> Sync for Pool<T> {}
unsafe impl<T: Default> Send for Pool<T> {}

pub trait PoolReturnable<T: Default> {
    fn clear_and_return_object(self, pool: &Pool<T>);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool() {
        let pool = Pool::<usize>::new();
        assert_eq!(pool.len(), 0);
        assert!(pool.is_empty());
        assert_eq!(pool.get_object(), 0);
        assert_eq!(pool.len(), 0);
        assert!(pool.is_empty());
        pool.return_object(1);
        assert_eq!(pool.len(), 1);
        assert!(!pool.is_empty());
        assert_eq!(pool.get_object(), 1);
        assert_eq!(pool.len(), 0);
        assert!(pool.is_empty());
    }
}
