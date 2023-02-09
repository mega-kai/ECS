use crate::component::*;
use std::{
    alloc::Layout,
    collections::HashMap,
    num::NonZeroUsize,
    ptr::{null, NonNull},
};

/// a type erased vector
struct TypeErasedVec {
    layout_of_component: Layout,
    data_heap_ptr: NonNull<u8>,
    len: usize,
    capacity: usize,
}
impl TypeErasedVec {
    /// does not handle ZST
    fn new<T>() -> Self {
        assert!(
            std::mem::size_of::<T>() != 0,
            "{} is a ZST",
            std::any::type_name::<T>()
        );

        let layout = Layout::new::<T>();
        let val = Self {
            layout_of_component: layout,
            data_heap_ptr: layout.dangling(),
            len: 0,
            capacity: 0,
        };
        val
    }

    fn with_capacity<T>(size: NonZeroUsize) -> Self {
        assert!(
            std::mem::size_of::<T>() != 0,
            "{} is a ZST",
            std::any::type_name::<T>()
        );

        let layout = Layout::new::<T>();
        let mut val = Self {
            layout_of_component: layout,
            data_heap_ptr: layout.dangling(),
            len: 0,
            capacity: 0,
        };
        val.grow_capacity(size);
        val
    }

    fn push(&mut self, ptr: NonNull<u8>, layout: Layout) {
        assert_eq!(self.layout_of_component, layout);

        if self.len >= self.capacity {
            //double the cap
            self.grow_capacity(unsafe { NonZeroUsize::new_unchecked(self.capacity) });
            self.insert_from_ptr(self.len - 1, ptr);
        } else {
            self.insert_from_ptr(self.len - 1, ptr);
        }
    }

    /// try to make sure this is only used to double the capacity
    fn grow_capacity(&mut self, grow: NonZeroUsize) {
        let new_capacity = self.capacity + grow.get();
        let (new_layout, _) = self
            .layout_of_component
            .repeat(new_capacity)
            .expect("could not repeat this layout");
        let new_data_ptr = if self.capacity == 0 {
            unsafe { std::alloc::alloc(new_layout) }
        } else {
            unsafe {
                std::alloc::realloc(
                    //starting at
                    self.data_heap_ptr.as_ptr(),
                    //the extent to uproot
                    self.layout_of_component
                        .repeat(self.capacity)
                        .expect("could not repeat layout")
                        .0,
                    //length of the new memory
                    new_layout.size(),
                )
            }
        };
        self.capacity = new_capacity;
        self.data_heap_ptr = unsafe { NonNull::new_unchecked(new_data_ptr) };
    }

    fn insert_from_ptr(&mut self, index: usize, ptr: NonNull<u8>) {
        self.len += 1;
    }

    fn get(&self) {}
    fn get_mut(&mut self) {}
    fn remove(&mut self, index: usize) {}
    fn swap(&mut self) {}
}

/// a type erased sparse set based consisted of a sparse vec and a dense vec
struct SparseSet {
    dense: TypeErasedVec,
    //usize == TypeErasedVec.index,
    sparse: Vec<usize>,
}
impl SparseSet {
    /// add a type erased component, returning its index value
    /// in the sparse vec
    fn add(&mut self, ptr: NonNull<u8>) -> usize {
        0
    }

    /// remove both the content in the sparse vec and the dense
    /// vec according to the index
    fn remove(&mut self, index: usize) {}

    /// get a pointer to that type erased component's address in the dense
    /// construct it back to a &C with the queried type layout
    fn get() -> *const u8 {
        null()
    }

    /// same with get() but ultimately returns a &mut C
    fn get_mut() -> *mut u8 {
        null::<u8>() as *mut _
    }
}

/// but in fact this interates mainly over the dense
/// vec rather than the whole set
// impl IntoIterator for SparseSet {
//     type Item = u8;

//     fn into_iter(self) -> Self::IntoIter {
//         todo!()
//     }

//     type IntoIter = u8;
// }

/// a hash map of ComponentIDs as keys and SparseSets as values
/// functioning as a central hub for adding/accessing stored components;
/// there will be a sparse set just for the entities, which are composed
/// of keys that can access it's components
pub struct Storage {
    // no repeating items
    data_hash: HashMap<ComponentID, SparseSet>,
}
impl Storage {
    pub fn new() -> Self {
        Self {
            data_hash: HashMap::new(),
        }
    }

    pub fn add_component<C: Component>(&mut self, component: C) -> usize {
        let id = component.id();
        let result = self.data_hash.get_mut(&id);
        if let Some(access) = result {
            access.add(NonNull::dangling());
            //add component
        } else {
            //create a new
        }
        0
    }

    /// this function is supposed to return an iterator of either &C, &mut C or C
    pub fn query<C: QueryIdentifier>(&mut self) -> Option<C> {
        todo!()
    }
}
