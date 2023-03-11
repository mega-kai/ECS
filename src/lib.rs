#![allow(dead_code, unused_variables, unused_imports, unused_mut)]
#![feature(alloc_layout_extra, map_try_insert)]
mod component;
mod scheduler;
mod storage;
use component::*;
use scheduler::*;
use storage::*;

pub struct ECS {
    storage: ComponentTable,
    scheduler: Scheduler,
}

impl ECS {
    pub fn new() -> Self {
        Self {
            storage: ComponentTable::new(),
            scheduler: Scheduler::new(),
        }
    }

    pub fn add_system(&mut self, system: System) {
        self.scheduler.add_system(system);
    }

    pub fn tick(&mut self) {
        self.scheduler.prepare_queue();
        for system in &self.scheduler.queue {
            system.run(&mut self.storage);
        }
    }
}

#[cfg(test)]
mod test {
    use std::alloc::Layout;

    use super::*;

    #[derive(Clone)]
    struct Health(i32);
    impl Component for Health {}

    #[derive(Clone, Debug)]
    struct Mana(i32);
    impl Component for Mana {}

    #[derive(Clone, Debug)]
    struct Player(&'static str);
    impl Component for Player {}

    fn take_comp<C0: Component, C1: Component>(c0: C0, c1: C1) {
        println!("{:?}", c0.id_instance());
        println!("{:?}", c1.id_instance());
        assert_ne!(c0.id_instance(), c1.id_instance());
    }

    // auto derive component trait
    type PLAYER = (Player, Health);

    #[test]
    fn main() {
        let player: PLAYER = (Player("name"), Health(100));
        take_comp(player.clone(), player);
        take_comp((Mana(12), Health(12)), (Health(12), Mana(12)));
    }

    #[test]
    fn type_erased_vec() {
        let vec_len = 4;
        let mut vec = TypeErasedColumn::new(Layout::new::<Player>(), vec_len);

        // add/get first one
        let mut player0 = Player("player 0");
        let index0 = vec.add((&mut player0 as *mut Player) as *mut u8);
        let ref0 = unsafe { vec.get(index0).unwrap().cast::<Player>().as_mut().unwrap() };
        assert_eq!(ref0.0, player0.0);

        // add/get second one
        let mut player1 = Player("player 1");
        let index1 = vec.add((&mut player1 as *mut Player) as *mut u8);
        let ref1 = unsafe { vec.get(index1).unwrap().cast::<Player>().as_mut().unwrap() };
        assert_eq!(ref1.0, player1.0);

        // remove the first one
        let ref_remove0 = unsafe {
            vec.remove(index0)
                .unwrap()
                .cast::<Player>()
                .as_mut()
                .unwrap()
        };
        assert_eq!(ref_remove0.0, player0.0);

        println!("-------------------------------------");
        println!(
            "the string removed from the vec at index {} is {}\n",
            index0, ref_remove0.0
        );
        println!(
            "the whole bit set is {:?}\n\nand only the second one should be occupied",
            vec.flags
        );
        println!("-------------------------------------");

        // add a third which should be at index 0
        let mut player2 = Player("player 2");
        let index2 = vec.add((&mut player2 as *mut Player) as *mut u8);
        let ref2 = unsafe { vec.get(index2).unwrap().cast::<Player>().as_mut().unwrap() };
        assert_eq!(ref2.0, player2.0);

        println!("-------------------------------------");
        println!(
            "the string removed from the vec at index {} is {}\n",
            index0, ref_remove0.0
        );
        println!(
            "the whole bit set is {:?}\n\nand first two should both be occupied",
            vec.flags
        );
        println!("-------------------------------------");

        // grow
        for i in 0..(vec_len - 2) {
            vec.add((&mut Player("filler") as *mut Player) as *mut u8);
        }

        println!("-------------------------------------\n");
        println!(
            "the whole bit set is {:?}\n\nand all should be occupied while {} should equal to {} which is the inital len",
            vec.flags,
            vec.flags.len(),
            vec_len
        );
        println!("-------------------------------------");

        vec.add((&mut Player("filler") as *mut Player) as *mut u8);

        println!("-------------------------------------\n");
        println!(
            "the whole bit set is {:?}\n\nand all should be occupied while {} should be equal to {} x 2 which is twice the inital len, same with {} which is the length of the dense vec",
            vec.flags,
            vec.flags.len(),
            vec_len,
            vec.capacity
        );
        println!("-------------------------------------");
        assert_eq!(vec.flags.len(), vec_len * 2);
        assert_eq!(vec.flags.len(), vec.capacity);

        // remove everything
        for i in 0..=5 {
            vec.remove(i).unwrap();
        }
        println!("-------------------------------------\n");
        println!(
            "the whole bit set is {:?}\n\nand all should be occupied while {} should be equal to {} x 2 which is twice the inital len, same with {} which is the length of the dense vec",
            vec.flags,
            vec.flags.len(),
            vec_len,
            vec.capacity
        );
        println!("-------------------------------------");
        assert_eq!(vec.flags.len(), vec_len * 2);
        assert_eq!(vec.flags.len(), vec.capacity);
    }

    #[test]
    fn storage() {
        let mut storage = ComponentTable::new();

        // add and retrieve
        let name0 = "pl 0";
        let player0 = Player(name0);
        let key0 = storage.add_component(player0);
        let ref0 = storage.get_as::<Player>(key0).unwrap();
        assert_eq!(ref0.0, name0);

        // remove
        let remove0 = storage.remove_as::<Player>(key0).unwrap();
        assert_eq!(remove0.0, name0);

        // error: mismatched types
        let name1 = "pl 1";
        let player1 = Player(name1);
        let key1 = storage.add_component(player1);
        let err1 = storage.get_as::<Mana>(key1);
        assert_eq!(err1.unwrap_err(), "generic and the key don't match");

        // error: wrong key index
        let name2 = "pl 2";
        let player2 = Player(name2);
        let key2 = storage.add_component(player2);
        let err2 = storage
            .get_as::<Player>(ComponentKey::new::<Player>(999))
            .unwrap_err();
        // println!("{}", err2);
        assert_eq!(err2, "index overflow in dense vec");

        // error: wrong key comp id
        let name3 = "pl 3";
        let player3 = Player(name3);
        let key3 = storage.add_component(player3);
        let err3 = storage
            .get_as::<Mana>(ComponentKey::new::<Mana>(999))
            .unwrap_err();
        // println!("{}", err3);
        assert_eq!(err3, "no such component type exist in this storage");
    }
}
