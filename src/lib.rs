#![allow(dead_code, unused_variables)]
#![feature(core_intrinsics, portable_simd)]
use std::{any::TypeId, fmt::Debug};

//for serde
trait Component: Debug + Copy + Clone + 'static {}

///sparse set for a single type of component
#[derive(Debug)]
struct SparseSet<T: Component> {
    //these usizes being the id for set of components
    sparse: Vec<usize>,
    //(!0, None)
    dense: Vec<(usize, Option<T>)>,
    type_id: TypeId,
}
impl<T: Component> SparseSet<T> {
    fn new<P: Component>() -> Self {
        SparseSet {
            //using !0 to represent empty
            sparse: vec![!0; 1],
            dense: vec![],
            type_id: TypeId::of::<P>(),
        }
    }

    fn add(&mut self, id: usize, data: T) {
        //dense must be init empty
        self.dense.push((id, Some(data)));

        let cap = self.sparse.capacity();
        //sparse must be at least one
        if id > cap - 1 {
            self.sparse.append(&mut vec![!0; cap + id]);
        }

        self.sparse[id] = self.dense.len() - 1;
    }

    fn get(&self, id: usize) -> Option<&T> {
        if let Some(&key) = self.sparse.get(id) {
            if let Some(thing) = self.dense.get(key) {
                thing.1.as_ref()
            } else {
                None
            }
        } else {
            None
        }
    }

    fn get_mut(&mut self, id: usize) -> Option<&mut T> {
        if let Some(&key) = self.sparse.get(id) {
            if let Some(thing) = self.dense.get_mut(key) {
                thing.1.as_mut()
            } else {
                None
            }
        } else {
            None
        }
    }

    fn remove(&mut self, id: usize) -> Option<T> {
        //make sure id is valid in sparse vec
        if let Some(&key) = self.sparse.get(id) {
            if let Some(thing) = self.dense.get_mut(key) {
                let stuff = thing.1;
                thing.1 = None;
                //what if stuff is none??
                stuff
            } else {
                None
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use std::simd::Simd;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    struct Test(i32);
    impl Test {
        fn add(&mut self) {
            self.0 += 1;
        }
    }
    impl Component for Test {}

    #[test]
    fn sparse_set() {
        let mut set: SparseSet<Test> = SparseSet::new::<Test>();
        set.add(0, Test(0));
        set.add(3, Test(3));
        let some = set.get(3);
        let none = set.get(2);
        //println!("{:?}", set);
        assert_eq!(some, Some(&Test(3)));
        assert_eq!(none, None);

        let mut_ref = set.get_mut(0);
        let thing = mut_ref.unwrap();
        thing.add();
        assert_eq!(thing, &mut Test(1));

        //you can borrow again even after a mut borrow
        let borrow_again = set.get(0);
        assert_eq!(borrow_again, Some(&Test(1)));
    }
}
