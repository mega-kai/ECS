use crate::component::*;
use crate::query::*;
use crate::scheduler::*;
use std::marker::PhantomData;
use std::{
    alloc::Layout,
    collections::HashMap,
    num::NonZeroUsize,
    ptr::{null, NonNull},
};

pub(crate) struct TypeErasedVec {
    layout_of_component: Layout,
    data_heap_ptr: *mut u8,
    len: usize,
    capacity: usize,
}
impl TypeErasedVec {
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
            self.grow_capacity(unsafe { NonZeroUsize::new_unchecked(self.capacity) });
            unsafe {
                self.raw_overwrite_at_ptr(self.len, ptr).unwrap();
            }
        } else {
            unsafe {
                self.raw_overwrite_at_ptr(self.len, ptr).unwrap();
            }
        }
    }

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
                    self.data_heap_ptr,
                    self.layout_of_component
                        .repeat(self.capacity)
                        .expect("could not repeat layout")
                        .0,
                    new_layout_of_whole_vec.size(),
                )
            }
        };
        self.capacity = new_capacity;
        self.data_heap_ptr = new_data_ptr;
    }

    pub(crate) unsafe fn raw_overwrite_at_ptr(
        &mut self,
        index: usize,
        src_ptr: *mut u8,
    ) -> Result<(), &str> {
        if index > self.len {
            Err("index overflow")
        } else {
            let raw_dst_ptr = self.data_heap_ptr.add(index * self.layout().size());
            std::ptr::copy(src_ptr, raw_dst_ptr, self.layout().size());
            self.len += 1;
            Ok(())
        }
    }

    pub(crate) unsafe fn get_ptr_from_index(&self, index: usize) -> Option<*mut u8> {
        if index > self.len {
            None
        } else {
            Some(self.data_heap_ptr.add(index * self.layout().size()))
        }
    }

    pub(crate) unsafe fn get_raw_ptr(&self, index: usize) -> Result<*mut u8, &str> {
        if index >= self.len {
            Err("index overflow in dense vec")
        } else {
            Ok(self.data_heap_ptr.add(index * self.layout().size()))
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

/// dense_vec[sparse_vec[comp_id]] = actual_comp_data
pub(crate) struct SparseSet {
    dense: TypeErasedVec,
    sparse: Vec<Option<usize>>,
    id_counter: usize,
}
impl SparseSet {
    pub(crate) fn new(layout: Layout) -> Self {
        Self {
            dense: TypeErasedVec::new(layout),
            sparse: vec![None; 64],
            id_counter: 0,
        }
    }

    pub(crate) fn add(&mut self, ptr: *mut u8) -> usize {
        // need to ensure length
        self.dense.push(ptr);
        self.sparse[self.id_counter] = Some(self.dense.len() - 1);
        self.id_counter += 1;
        // issuing a new comp id number
        self.id_counter - 1
    }

    pub(crate) fn get(&self, index: usize) -> Result<*mut u8, &str> {
        unsafe {
            let sparse_result = self.sparse[index].ok_or("index is empty")?;
            let dense_result = self.dense.get_raw_ptr(sparse_result)?;
            Ok(dense_result)
        }
    }

    pub(crate) fn remove(&mut self, index: usize) -> Result<*mut u8, &str> {
        unsafe {
            let sparse_result = self.sparse[index].ok_or("index is empty")?;
            self.sparse[index] = None;
            let dense_result = self.dense.get_raw_ptr(sparse_result)?;
            Ok(dense_result)
        }
    }
}

pub struct Storage {
    data_hash: HashMap<ComponentID, SparseSet>,
}

impl Storage {
    pub(crate) fn new() -> Self {
        Self {
            data_hash: HashMap::new(),
        }
    }

    pub(crate) fn try_gaining_access(&mut self, id: ComponentID) -> Result<&mut SparseSet, &str> {
        self.data_hash
            .get_mut(&id)
            .ok_or("no such type of component stored")
    }

    pub(crate) fn init_set<C: Component>(&mut self) -> &mut SparseSet {
        let id = ComponentID::new::<C>();
        let layout = Layout::new::<C>();
        self.data_hash.insert(id, SparseSet::new(layout));
        self.data_hash.get_mut(&id).unwrap()
    }

    pub(crate) fn add_component<C: Component>(&mut self, mut component: C) -> ComponentKey {
        match self.try_gaining_access(C::id()) {
            Ok(val) => ComponentKey::new::<C>(val.add((&mut component as *mut C).cast::<u8>())),
            Err(_) => ComponentKey::new::<C>(
                self.init_set::<C>()
                    .add((&mut component as *mut C).cast::<u8>()),
            ),
        }
    }

    // maybe a swap function??

    pub(crate) fn get<C: Component>(&mut self, key: ComponentKey) -> Result<&mut C, &str> {
        if C::id() != key.id() {
            return Err("generic and the key don't match");
        }
        let access = self.try_gaining_access(key.id())?;
        unsafe { Ok(access.get(key.index())?.cast::<C>().as_mut().unwrap()) }
    }

    pub(crate) fn remove<C: Component>(&mut self, key: ComponentKey) -> Result<C, &str> {
        unsafe {
            self.try_gaining_access(key.id())?
                .get(key.index())?
                .cast::<C>()
                .as_mut()
                .cloned()
                .ok_or("err")
        }
    }
}
