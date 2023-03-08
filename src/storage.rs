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
    fn get_first(&self, target: <Self as VecHelperFunc>::Target) -> Option<usize>;
    fn double_cap(&mut self);
}

impl VecHelperFunc for Vec<Status> {
    type Target = Status;

    fn get_first(&self, target: <Self as VecHelperFunc>::Target) -> Option<usize> {
        for (index, val) in self.iter().enumerate() {
            if val == &target {
                return Some(index);
            }
        }
        None
    }

    fn double_cap(&mut self) {
        self.resize(self.len() * 2, Status::Empty);
    }
}

#[derive(Clone, PartialEq, Debug)]
pub(crate) enum Status {
    Occupied,
    Empty,
}

pub(crate) struct TypeErasedVec {
    layout_of_component: Layout,
    data_heap_ptr: *mut u8,
    pub(crate) capacity: usize,

    // one to one correspondence, basically a bit set
    pub(crate) flags: Vec<Status>,
}
impl TypeErasedVec {
    pub(crate) fn new(layout: Layout, size: usize) -> Self {
        assert!(layout.size() != 0, "type is a ZST",);

        let data_heap_ptr = unsafe { std::alloc::alloc(layout.repeat(size).unwrap().0) };
        Self {
            layout_of_component: layout,
            data_heap_ptr,
            capacity: size,
            flags: vec![Status::Empty; size],
        }
    }

    // add to the first entry that is None
    pub(crate) fn add(&mut self, ptr: *mut u8) -> usize {
        let index = self.flags.get_first(Status::Empty).unwrap_or_else(|| {
            let len = self.flags.len();
            self.double_cap();
            self.flags.double_cap();
            len
        });
        self.flags[index] = Status::Occupied;
        unsafe {
            let raw_dst_ptr = self
                .data_heap_ptr
                .add(index * self.layout_of_component.size());
            std::ptr::copy(ptr, raw_dst_ptr, self.layout_of_component.size());
        }
        index
    }

    pub(crate) fn get(&self, index: usize) -> Result<*mut u8, &'static str> {
        if index >= self.capacity {
            Err("index overflow in dense vec")
        } else {
            unsafe {
                Ok(self
                    .data_heap_ptr
                    .add(index * self.layout_of_component.size()))
            }
        }
    }

    pub(crate) fn remove(&mut self, index: usize) -> Result<*mut u8, &'static str> {
        if index >= self.capacity {
            Err("index overflow in dense vec")
        } else {
            self.flags[index] = Status::Empty;
            unsafe {
                Ok(self
                    .data_heap_ptr
                    .add(index * self.layout_of_component.size()))
            }
        }
    }

    fn double_cap(&mut self) {
        let new_capacity = self.capacity * 2;
        let (new_layout_of_whole_vec, _) = self
            .layout_of_component
            .repeat(new_capacity)
            .expect("could not repeat this layout");
        let new_data_ptr = unsafe {
            std::alloc::realloc(
                self.data_heap_ptr,
                self.layout_of_component
                    .repeat(self.capacity)
                    .expect("could not repeat layout")
                    .0,
                new_layout_of_whole_vec.size(),
            )
        };
        self.capacity = new_capacity;
        self.data_heap_ptr = new_data_ptr;
    }
}

pub struct Storage {
    data_hash: HashMap<ComponentID, TypeErasedVec>,
}

impl Storage {
    pub(crate) fn new() -> Self {
        Self {
            data_hash: HashMap::new(),
        }
    }

    fn ensure_access<C: Component>(&mut self) -> &mut TypeErasedVec {
        self.data_hash
            .entry(C::id())
            .or_insert(TypeErasedVec::new(Layout::new::<C>(), 64))
    }

    fn try_access<C: Component>(&mut self) -> Result<&mut TypeErasedVec, &'static str> {
        if let Some(access) = self.data_hash.get_mut(&C::id()) {
            Ok(access)
        } else {
            Err("no such component type exist in this storage")
        }
    }

    pub(crate) fn add_component<C: Component>(&mut self, mut component: C) -> ComponentKey {
        assert!(component.id_instance() == C::id(), "type inconsistent");
        ComponentKey::new::<C>(
            self.ensure_access::<C>()
                .add((&mut component as *mut C).cast::<u8>()),
        )
    }

    pub(crate) fn get<C: Component>(&mut self, key: ComponentKey) -> Result<&mut C, &'static str> {
        if C::id() != key.id() {
            return Err("generic and the key don't match");
        }
        let access = self.try_access::<C>()?;
        unsafe { Ok(access.get(key.index())?.cast::<C>().as_mut().unwrap()) }
    }

    pub(crate) fn remove<C: Component>(&mut self, key: ComponentKey) -> Result<C, &'static str> {
        unsafe {
            Ok(self
                .try_access::<C>()?
                .remove(key.index)
                .unwrap()
                .cast::<C>()
                .as_mut()
                .cloned()
                .unwrap())
        }
    }
}

// /// dense_vec[sparse_vec[comp_id]] = actual_comp_data
// pub(crate) struct SparseSet {
//     dense: TypeErasedVec,
//     sparse: Vec<Option<usize>>,
// }
// impl SparseSet {
//     pub(crate) fn new(layout: Layout) -> Self {
//         Self {
//             dense: TypeErasedVec::new(layout, 64),
//             sparse: vec![None; 64],
//         }
//     }

//     pub(crate) fn write(&mut self, ptr: *mut u8) -> usize {
//         // need to ensure length
//         let index = self.sparse.get_first(&None).unwrap();
//         // dense is not being reused
//         self.dense.push(ptr);
//         self.sparse[index] = Some(self.dense.len() - 1);
//         index
//     }

//     pub(crate) fn get(&self, index: usize) -> Result<*mut u8, &'static str> {
//         unsafe {
//             let sparse_result = self.sparse[index].ok_or("index is empty")?;
//             let dense_result = self.dense.get(sparse_result)?;
//             Ok(dense_result)
//         }
//     }

//     pub(crate) fn remove(&mut self, index: usize) -> Result<*mut u8, &'static str> {
//         unsafe {
//             let sparse_result = self.sparse[index].ok_or("index is empty")?;
//             let dense_result = self.dense.get(sparse_result)?;
//             self.sparse[index] = None;
//             Ok(dense_result)
//         }
//     }
// }
