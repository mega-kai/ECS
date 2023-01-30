#![allow(dead_code, unused_variables)]
#![feature(core_intrinsics, portable_simd)]
use std::{any::TypeId, fmt::Debug};

//for serde
trait Component: Debug + Copy + Clone + 'static {}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
struct Key {
    index: usize,
    mut_ref_issued: bool,
    generation: usize,
}
impl Key {
    fn new_uninit() -> Self {
        Self {
            index: !0,
            mut_ref_issued: false,
            generation: 0,
        }
    }
    fn new(index: usize) -> Self {
        Self {
            index,
            mut_ref_issued: false,
            generation: 0,
        }
    }
}

/// sparse set for a single type of component;
/// this is a data structure that don't prevent data racing;
/// you can issue multiple mutable references with external mechanics;
/// it's also one sided, meaning only sparse points to dense;
#[derive(Debug)]
struct SparseSet<T: Component> {
    //these usizes being the id for set of components
    sparse: Vec<Key>,
    //(Key::new(!0), None)
    dense: Vec<Option<T>>,
    type_id: TypeId,
}
impl<T: Component> SparseSet<T> {
    fn new<P: Component>() -> Self {
        SparseSet {
            //using !0 to represent empty
            sparse: vec![Key::new_uninit()],
            dense: vec![],
            type_id: TypeId::of::<P>(),
        }
    }

    fn add(&mut self, id: usize, data: T) -> Key {
        let key = Key::new(id);

        //dense must be init empty
        self.dense.push(Some(data));

        let cap = self.sparse.capacity();
        //sparse must be at least one
        if key.index > cap - 1 {
            self.sparse
                .append(&mut vec![Key::new_uninit(); cap + key.index]);
        }
        self.sparse[key.index] = key;

        key
    }

    fn get(&self, id: usize) -> Option<&T> {
        if let Some(key) = self.sparse.get(id) {
            if let Some(thing) = self.dense.get(key.index) {
                thing.as_ref()
            } else {
                None
            }
        } else {
            None
        }
    }

    fn get_mut(&mut self, key: Key) -> Option<&mut T> {
        if let Some(result) = self.sparse.get(key.index) {
            if let Some(thing) = self.dense.get_mut(result.index) {
                thing.as_mut()
            } else {
                None
            }
        } else {
            None
        }
    }

    fn remove(&mut self, id: usize) -> Option<T> {
        //make sure id is valid in sparse vec
        if let Some(key) = self.sparse.get(id) {
            if let Some(thing) = self.dense.get_mut(key.index) {
                let stuff = thing.clone();
                *thing = None;
                //what if stuff is none??
                *stuff
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
        set.add(Key::new(0), Test(0));
        set.add(Key::new(3), Test(3));
        let some = set.get(3);
        let none = set.get(2);
        //println!("{:?}", set);
        assert_eq!(some, Some(&Test(3)));
        assert_eq!(none, None);

        // you can issue two mutable references
        //let ref1 = set.get_mut(0);
        //let test1 = ref1.unwrap();
        //test1.add();
        //let ref2 = set.get_mut(0);
        //let test2 = ref2.unwrap();
        //test2.add();

        let mut vec: Vec<i32> = vec![];
        let refe1 = vec.get_mut(0).unwrap();
        let refe2 = vec.get_mut(0).unwrap();

        assert_eq!(set.get(0), Some(&Test(2)));
    }
}
