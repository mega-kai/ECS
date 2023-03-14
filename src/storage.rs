use crate::component::*;
use crate::scheduler::*;
use std::collections::HashSet;
use std::{alloc::Layout, collections::HashMap};

pub(crate) struct TypeErasedColumn {
    layout_of_component: Layout,
    data_heap_ptr: *mut u8,
    pub(crate) capacity: usize,
    pub(crate) len: usize,
    pub(crate) sparse: Vec<Option<usize>>,
}
impl TypeErasedColumn {
    pub(crate) fn query_all_dense_ptr_with_sparse_entity_id(&self) -> Vec<(usize, *mut u8)> {
        let mut result_vec: Vec<(usize, *mut u8)> = vec![];
        for val in self.sparse.iter() {
            if let Some(current_id) = val {
                result_vec.push((*current_id, unsafe { self.get(*current_id).unwrap() }));
            } else {
                continue;
            }
        }
        result_vec
    }

    pub(crate) fn new(layout: Layout, size: usize) -> Self {
        assert!(layout.size() != 0, "type is a ZST",);

        let data_heap_ptr = unsafe { std::alloc::alloc(layout.repeat(size).unwrap().0) };
        Self {
            layout_of_component: layout,
            data_heap_ptr,
            capacity: size,
            len: 0,
            sparse: vec![None; size],
        }
    }

    pub(crate) fn add(&mut self, ptr: *mut u8, entity_id: usize) -> *mut u8 {
        if entity_id >= self.sparse.len() {
            self.sparse.resize(entity_id + 1, None);
        }

        if self.len >= self.capacity {
            self.double_dense_cap();
        }

        self.sparse[entity_id] = Some(self.len);

        unsafe {
            let raw_dst_ptr = self
                .data_heap_ptr
                .add(self.len * self.layout_of_component.size());
            std::ptr::copy(ptr, raw_dst_ptr, self.layout_of_component.size());
            raw_dst_ptr
        }
    }

    pub(crate) unsafe fn get(&self, entity_id: usize) -> Option<*mut u8> {
        if entity_id >= self.sparse.len() {
            None
        } else {
            Some(
                self.data_heap_ptr
                    .add(entity_id * self.layout_of_component.size()),
            )
        }
    }

    pub(crate) unsafe fn remove(&mut self, entity_id: usize) -> Option<*mut u8> {
        if entity_id >= self.sparse.len() {
            None
        } else {
            let num = self.sparse[entity_id].unwrap();
            self.sparse[entity_id] = None;
            Some(
                self.data_heap_ptr
                    .add(num * self.layout_of_component.size()),
            )
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
    current_entity_id: usize,
}

impl ComponentTable {
    pub(crate) fn new() -> Self {
        Self {
            data_hash: HashMap::new(),
            // 0 would be reserved
            current_entity_id: 0,
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
        self.current_entity_id += 1;
        let new_entity_id = self.current_entity_id;
        let dst_ptr = self
            .ensure_access_of_type::<C>()
            .add((&mut component as *mut C).cast::<u8>(), new_entity_id - 1);
        ComponentAccess::new(new_entity_id - 1, ComponentID::new::<C>(), dst_ptr)
    }

    pub(crate) fn link(&mut self, key1: ComponentAccess, key2: ComponentAccess) {
        todo!()
    }

    pub(crate) fn get_as<C: Component>(
        &mut self,
        key: ComponentAccess,
    ) -> Result<&mut C, &'static str> {
        if C::id() != key.id {
            return Err("generic and the key don't match");
        }
        let access = self.try_access::<C>()?;
        unsafe {
            Ok(access
                .get(key.entity_id)
                .unwrap()
                .cast::<C>()
                .as_mut()
                .unwrap())
        }
    }

    pub(crate) fn remove_as<C: Component>(
        &mut self,
        key: ComponentAccess,
    ) -> Result<C, &'static str> {
        unsafe {
            Ok(self
                .try_access::<C>()?
                .remove(key.entity_id)
                .unwrap()
                .cast::<C>()
                .as_mut()
                .cloned()
                .unwrap())
        }
    }

    pub(crate) fn query_single_from_type<C: Component>(&self) -> Vec<ComponentAccess> {
        if let Some(access) = self.data_hash.get(&C::id()) {
            let mut raw_vec = access.query_all_dense_ptr_with_sparse_entity_id();
            let mut result_access_vec: Vec<ComponentAccess> = vec![];
            for (index, ptr) in raw_vec {
                result_access_vec.push(ComponentAccess::new(index, ComponentID::new::<C>(), ptr));
            }
            result_access_vec
        } else {
            panic!("no such component type within the table")
        }
    }

    fn query_related_from_index(&self, entity_id: usize) -> Vec<ComponentAccess> {
        let mut vec: Vec<ComponentAccess> = vec![];
        for (id, column) in self.data_hash.iter() {
            if let Some(ptr) = unsafe { column.get(entity_id) } {
                vec.push(ComponentAccess::new(entity_id, *id, ptr));
            } else {
                continue;
            }
        }
        vec
    }

    pub(crate) fn query_related_with_key(&self, key: ComponentAccess) -> Vec<ComponentAccess> {
        let index = key.entity_id;
        self.query_related_from_index(index)
    }
}
