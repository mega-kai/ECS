//type erased storage
use crate::component::*;
use crate::query::*;
use crate::scheduler::*;
use std::{
    alloc::Layout,
    collections::HashMap,
    num::NonZeroUsize,
    ptr::{null, NonNull},
};

/// a type erased vector
pub(crate) struct TypeErasedVec {
    layout_of_component: Layout,
    data_heap_ptr: *mut u8,
    len: usize,
    capacity: usize,
}
impl TypeErasedVec {
    /// does not handle ZST
    pub(crate) fn new(layout: Layout) -> Self {
        assert!(layout.size() != 0, "type is a ZST",);

        let mut val = Self {
            layout_of_component: layout,
            data_heap_ptr: layout.dangling().as_ptr(),
            len: 0,
            capacity: 0,
        };
        val.grow_capacity(unsafe { NonZeroUsize::new_unchecked(64) });
        val
    }

    pub(crate) fn push(&mut self, ptr: *mut u8) {
        if self.len >= self.capacity {
            //double the cap
            self.grow_capacity(unsafe { NonZeroUsize::new_unchecked(self.capacity) });
            unsafe {
                self.raw_insert_from_ptr(self.len, ptr);
            }
        } else {
            unsafe {
                self.raw_insert_from_ptr(self.len, ptr);
            }
        }
    }

    /// try to make sure this is only used to double the capacity
    pub(crate) fn grow_capacity(&mut self, grow: NonZeroUsize) {
        let new_capacity = self.capacity + grow.get();
        let (new_layout_of_whole_vec, _) = self
            .layout_of_component
            .repeat(new_capacity)
            .expect("could not repeat this layout");
        let new_data_ptr = if self.capacity == 0 {
            unsafe { std::alloc::alloc(new_layout_of_whole_vec) }
        } else {
            unsafe {
                std::alloc::realloc(
                    //starting at
                    self.data_heap_ptr,
                    //the extent to uproot
                    self.layout_of_component
                        .repeat(self.capacity)
                        .expect("could not repeat layout")
                        .0,
                    //length of the new memory
                    new_layout_of_whole_vec.size(),
                )
            }
        };
        self.capacity = new_capacity;
        self.data_heap_ptr = new_data_ptr;
    }

    /// will overwrite the value if the index is taken
    pub(crate) unsafe fn raw_insert_from_ptr(&mut self, index: usize, src_ptr: *mut u8) {
        //check if index is valid
        let raw_dst_ptr = self.get_ptr_from_index(index).expect("index overflow");
        //memmove size() amount of bytes
        std::ptr::copy(src_ptr, raw_dst_ptr, self.layout().size());
        //incrememt the len
        self.len += 1;
    }

    pub(crate) unsafe fn get_ptr_from_index(&self, index: usize) -> Option<*mut u8> {
        if index > self.len {
            None
        } else {
            Some(self.data_heap_ptr.add(index * self.layout().size()))
        }
    }

    /// different from get_ptr_from_index as this is not for insertion,
    /// this is for access
    pub(crate) unsafe fn get(&self, index: usize) -> Option<*mut u8> {
        if index >= self.len {
            None
        } else {
            Some(self.data_heap_ptr.add(index * self.layout().size()))
        }
    }

    pub(crate) fn layout(&self) -> Layout {
        self.layout_of_component
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn cap(&self) -> usize {
        self.capacity
    }
}

/// a type erased sparse set based consisted of a sparse vec and a dense vec
pub(crate) struct SparseSet {
    dense: TypeErasedVec,
    //the usize is the index for dense, the index of sparse is the value from the key
    sparse: Vec<Option<usize>>,
}
impl SparseSet {
    pub(crate) fn new(layout: Layout) -> Self {
        Self {
            dense: TypeErasedVec::new(layout),
            sparse: vec![None; 64],
        }
    }

    /// add a type erased component, returning its index value
    /// in the sparse vec
    fn add(&mut self, ptr: *mut u8) -> usize {
        0
    }

    /// remove both the content in the sparse vec and the dense
    /// vec according to the index
    fn remove(&mut self, index: usize) -> *mut u8 {
        todo!()
    }

    /// get a pointer to that type erased component's address in the dense
    /// construct it back to a &C with the queried type layout
    fn get() -> *mut u8 {
        todo!()
    }
}

/// a hash map of ComponentIDs as keys and SparseSets as values
/// functioning as a central hub for adding/accessing stored components;
/// there will be a sparse set just for the entities, which are composed
/// of keys that can access it's components
pub struct Storage {
    // no repeating items
    data_hash: HashMap<ComponentID, SparseSet>,
}
impl Storage {
    pub(crate) fn new() -> Self {
        Self {
            data_hash: HashMap::new(),
        }
    }

    pub(crate) fn add_component<C: Component>(&mut self, mut component: C) -> ComponentKey {
        let id = component.id();
        if self.check_id_validity(id) {
            let access = self.data_hash.get_mut(&id).unwrap();
            let ptr = (&mut component as *mut C).cast::<u8>();
            return ComponentKey::new::<C>(access.add(ptr));
        } else {
            let access = self.init_set::<C>();
            let ptr = (&mut component as *mut C).cast::<u8>();
            return ComponentKey::new::<C>(access.add(ptr));
        }
    }

    pub(crate) fn check_id_validity(&mut self, id: ComponentID) -> bool {
        self.data_hash.get_mut(&id).is_some()
    }

    pub(crate) fn init_set<C: Component>(&mut self) -> &mut SparseSet {
        let id = ComponentID::new::<C>();
        let layout = Layout::new::<C>();
        self.data_hash.insert(id, SparseSet::new(layout));
        self.data_hash.get_mut(&id).unwrap()
    }
}
