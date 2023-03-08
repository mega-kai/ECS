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

trait VecHelperFunc {
    type Target;
    fn get_first(&self, target: &<Self as VecHelperFunc>::Target) -> Option<usize>;
    fn double_cap(&mut self);
}

impl<T: PartialEq> VecHelperFunc for Vec<T> {
    type Target = T;

    fn get_first(&self, target: &<Self as VecHelperFunc>::Target) -> Option<usize> {
        for (index, val) in self.iter().enumerate() {
            if val == target {
                return Some(index);
            }
        }
        None
    }

    fn double_cap(&mut self) {
        todo!()
    }
}

pub(crate) struct TypeErasedVec {
    layout_of_component: Layout,
    data_heap_ptr: *mut u8,
    len: usize,
    capacity: usize,
}
impl TypeErasedVec {
    pub(crate) fn new(layout: Layout, size: usize) -> Self {
        assert!(layout.size() != 0, "type is a ZST",);

        let data_heap_ptr = unsafe { std::alloc::alloc(layout.repeat(size).unwrap().0) };
        Self {
            layout_of_component: layout,
            data_heap_ptr,
            len: 0,
            capacity: size,
        }
    }

    pub(crate) fn push(&mut self, ptr: *mut u8) {
        if self.len >= self.capacity {
            self.double_cap();
            unsafe {
                self.overwrite(self.len, ptr).unwrap();
            }
        } else {
            unsafe {
                self.overwrite(self.len, ptr).unwrap();
            }
        }
    }

    pub(crate) fn double_cap(&mut self) {
        let new_capacity = self.capacity * 2;
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

    pub(crate) unsafe fn overwrite(&mut self, index: usize, src_ptr: *mut u8) -> Result<(), &str> {
        if index > self.len {
            Err("index overflow")
        } else {
            let raw_dst_ptr = self.data_heap_ptr.add(index * self.layout().size());
            std::ptr::copy(src_ptr, raw_dst_ptr, self.layout().size());
            self.len += 1;
            Ok(())
        }
    }

    pub(crate) unsafe fn get(&self, index: usize) -> Result<*mut u8, &str> {
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
}
impl SparseSet {
    pub(crate) fn new(layout: Layout) -> Self {
        Self {
            dense: TypeErasedVec::new(layout, 64),
            sparse: vec![None; 64],
        }
    }

    pub(crate) fn add(&mut self, ptr: *mut u8) -> usize {
        // need to ensure length
        let index = self.sparse.get_first(&None).unwrap();
        self.dense.push(ptr);
        self.sparse[index] = Some(self.dense.len() - 1);
        index
    }

    pub(crate) fn get(&self, index: usize) -> Result<*mut u8, &str> {
        unsafe {
            let sparse_result = self.sparse[index].ok_or("index is empty")?;
            let dense_result = self.dense.get(sparse_result)?;
            Ok(dense_result)
        }
    }

    pub(crate) fn remove(&mut self, index: usize) -> Result<*mut u8, &str> {
        unsafe {
            let sparse_result = self.sparse[index].ok_or("index is empty")?;
            let dense_result = self.dense.get(sparse_result)?;
            self.sparse[index] = None;
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

    pub(crate) fn ensure_access<C: Component>(&mut self) -> &mut SparseSet {
        self.data_hash
            .entry(C::id())
            .or_insert(SparseSet::new(Layout::new::<C>()))
    }

    pub(crate) fn try_access<C: Component>(&mut self) -> Result<&mut SparseSet, &str> {
        if let Some(access) = self.data_hash.get_mut(&C::id()) {
            Ok(access)
        } else {
            Err("no such component type exist in this storage")
        }
    }

    pub(crate) fn add_component<C: Component>(&mut self, mut component: C) -> ComponentKey {
        let num = self
            .ensure_access::<C>()
            .add((&mut component as *mut C).cast::<u8>());
        ComponentKey::new::<C>(num)
    }

    pub(crate) fn get<C: Component>(&mut self, key: ComponentKey) -> Result<&mut C, &str> {
        if C::id() != key.id() {
            return Err("generic and the key don't match");
        }
        let access = self.try_access::<C>()?;
        unsafe { Ok(access.get(key.index())?.cast::<C>().as_mut().unwrap()) }
    }

    pub(crate) fn remove<C: Component>(&mut self, key: ComponentKey) -> Result<C, &str> {
        unsafe {
            self.try_access::<C>()?
                .get(key.index())?
                .cast::<C>()
                .as_mut()
                .cloned()
                .ok_or("err")
        }
    }
}
