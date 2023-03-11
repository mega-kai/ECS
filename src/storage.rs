use crate::component::*;
use crate::scheduler::*;
use std::collections::HashSet;
use std::{alloc::Layout, collections::HashMap};

trait VecHelperFunc {
    type Target;
    fn get_first(&self, target: <Self as VecHelperFunc>::Target) -> Option<usize>;
    fn double_cap(&mut self);
}

impl VecHelperFunc for Vec<Status> {
    type Target = Status;

    fn get_first(&self, target: Self::Target) -> Option<usize> {
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

pub(crate) struct TypeErasedColumn {
    layout_of_component: Layout,
    data_heap_ptr: *mut u8,
    pub(crate) capacity: usize,
    pub(crate) flags: Vec<Status>,
}
impl TypeErasedColumn {
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

/// comp keys with same index are considered related;
/// columns are type and rows are index
pub struct ComponentTable {
    data_hash: HashMap<ComponentID, TypeErasedColumn>,
    row_freehead: usize,
    column_freehead: usize,
}

impl ComponentTable {
    pub(crate) fn new() -> Self {
        Self {
            data_hash: HashMap::new(),
            column_freehead: 0,
            row_freehead: 0,
        }
    }

    fn ensure_access<C: Component>(&mut self) -> &mut TypeErasedColumn {
        self.data_hash
            .entry(C::id())
            .or_insert(TypeErasedColumn::new(Layout::new::<C>(), 64))
    }

    fn try_access<C: Component>(&mut self) -> Result<&mut TypeErasedColumn, &'static str> {
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

    pub(crate) fn get_as<C: Component>(
        &mut self,
        key: ComponentKey,
    ) -> Result<&mut C, &'static str> {
        if C::id() != key.ty {
            return Err("generic and the key don't match");
        }
        let access = self.try_access::<C>()?;
        unsafe { Ok(access.get(key.index)?.cast::<C>().as_mut().unwrap()) }
    }

    pub(crate) fn remove_as<C: Component>(&mut self, key: ComponentKey) -> Result<C, &'static str> {
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

    pub(crate) fn query_single<C: Component>(&mut self) -> Vec<&mut C> {
        if let Some(val) = self.data_hash.get_mut(&C::id()) {
            todo!()
        } else {
            vec![]
        }
    }

    pub(crate) fn get_associated_comps(
        &self,
        key: ComponentKey,
    ) -> Result<Vec<ComponentKey>, &'static str> {
        // self.graph_data.get(&key);
        todo!()
    }
}
