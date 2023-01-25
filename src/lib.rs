#![allow(dead_code, unused_variables)]
#![feature(core_intrinsics)]
use std::{any::TypeId, fmt::Debug};

//for serde
trait Component: Debug + Copy + Clone + 'static {}

///sparse set for a single type of component
#[derive(Debug)]
struct SparseSet<T: Component> {
    //these usizes being the id for set of components
    sparse: Vec<usize>,
    dense: Vec<(usize, Option<T>)>,
    type_id: TypeId,
}
impl<T: Component> SparseSet<T> {
    fn new() -> Self {
        SparseSet {
            //using !0 to represent empty
            sparse: vec![],
            dense: vec![],
            type_id: TypeId::of::<T>(),
        }
    }

    /// make sure that id wouldn't be duped
    fn add(&mut self, id: usize, data: T) {
        //push the actual data in dense array
        self.dense.push((id, Some(data)));

        //check if the id exceeds the cap
        let cap = self.sparse.capacity();
        if id > cap - 1 {
            //dupe the cap and populate with !0
            self.sparse.append(&mut vec![!0; cap]);
        }

        self.sparse[id] = self.dense.len() - 1;
    }

    fn get(&self, id: usize) -> Option<&T> {
        if let Some(&key) = self.sparse.get(id) {
            todo!()
        } else {
            None
        }
    }

    fn get_mut(&mut self, id: usize) -> Option<&mut T> {
        if let Some(&key) = self.sparse.get(id) {
            todo!()
        } else {
            None
        }
    }

    fn remove(&mut self, id: usize) -> Option<T> {
        if let Some(&key) = self.sparse.get(id) {
            todo!()
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Test(u32);
impl Component for Test {}
