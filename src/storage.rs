use crate::component::*;
use crate::scheduler::*;
use std::collections::HashSet;
use std::{alloc::Layout, collections::HashMap};

pub(crate) struct TypeErasedColumn {
    layout_of_component: Layout,
    data_heap_ptr: *mut u8,
    pub(crate) capacity: usize,
    pub(crate) len: usize,
    pub(crate) sparse: Vec<usize>,
}
impl TypeErasedColumn {
    pub(crate) fn query_all_valid_data(&mut self) {}

    pub(crate) fn new(layout: Layout, size: usize) -> Self {
        assert!(layout.size() != 0, "type is a ZST",);

        let data_heap_ptr = unsafe { std::alloc::alloc(layout.repeat(size).unwrap().0) };
        Self {
            layout_of_component: layout,
            data_heap_ptr,
            capacity: size,
            len: 0,
            sparse: vec![0; size],
        }
    }

    pub(crate) fn add(&mut self, ptr: *mut u8, row_index: usize) {
        // recursively doing so
        if row_index >= self.capacity {
            self.double_dense_cap();
            self.sparse.resize(self.sparse.len() * 2, 0);
        }

        self.sparse[row_index] = 0;
        unsafe {
            let raw_dst_ptr = self
                .data_heap_ptr
                .add(row_index * self.layout_of_component.size());
            std::ptr::copy(ptr, raw_dst_ptr, self.layout_of_component.size());
        }
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
            self.sparse[index] = 0;
            unsafe {
                Ok(self
                    .data_heap_ptr
                    .add(index * self.layout_of_component.size()))
            }
        }
    }

    fn double_dense_cap(&mut self) {
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

pub struct ComponentTable {
    data_hash: HashMap<ComponentID, TypeErasedColumn>,
    comp_group_index: usize,
}

impl ComponentTable {
    pub(crate) fn new() -> Self {
        Self {
            data_hash: HashMap::new(),
            comp_group_index: 0,
        }
    }

    fn ensure_access_of_type<C: Component>(&mut self) -> &mut TypeErasedColumn {
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

    pub(crate) fn insert<C: Component>(&mut self, mut component: C) -> ComponentAccess {
        self.comp_group_index += 1;
        let row_index = self.comp_group_index;
        self.ensure_access_of_type::<C>()
            .add((&mut component as *mut C).cast::<u8>(), row_index - 1);
        ComponentAccess::new_from_type::<C>(row_index - 1)
    }

    pub(crate) fn link(&mut self, key1: ComponentAccess, key2: ComponentAccess) {
        todo!()
    }

    pub(crate) fn get_as<C: Component>(
        &mut self,
        key: ComponentAccess,
    ) -> Result<&mut C, &'static str> {
        if C::id() != key.ty {
            return Err("generic and the key don't match");
        }
        let access = self.try_access::<C>()?;
        unsafe { Ok(access.get(key.row_index)?.cast::<C>().as_mut().unwrap()) }
    }

    pub(crate) fn remove_as<C: Component>(
        &mut self,
        key: ComponentAccess,
    ) -> Result<C, &'static str> {
        unsafe {
            Ok(self
                .try_access::<C>()?
                .remove(key.row_index)
                .unwrap()
                .cast::<C>()
                .as_mut()
                .cloned()
                .unwrap())
        }
    }

    pub(crate) fn query_single<C: Component>(&self) -> Vec<&mut C> {
        if let Some(access) = self.data_hash.get(&C::id()) {
            todo!()
        } else {
            panic!("no such component type within the table")
        }
    }

    fn query_related_with_index(&self, index: usize) -> Vec<ComponentAccess> {
        let mut vec: Vec<ComponentAccess> = vec![];
        for (id, column) in self.data_hash.iter() {
            if let Ok(ptr) = column.get(index) {
                vec.push(ComponentAccess::new(index, *id, ptr));
            } else {
                continue;
            }
        }
        vec
    }

    pub(crate) fn query_related_with_key(&mut self, key: ComponentAccess) -> Vec<ComponentAccess> {
        let index = key.row_index;
        self.query_related_with_index(index)
    }
}
