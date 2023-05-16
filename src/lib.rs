#![allow(
    dead_code,
    unused_variables,
    unreachable_code,
    unused_mut,
    unused_assignments
)]
#![feature(
    alloc_layout_extra,
    map_try_insert,
    core_intrinsics,
    const_trait_impl,
    const_type_id,
    const_mut_refs,
    const_type_name,
    portable_simd,
    array_zip,
    slice_as_chunks
)]

use core::panic;
use hashbrown::HashMap;
use std::alloc::{alloc, dealloc, realloc};
use std::intrinsics::type_id;
use std::marker::PhantomData;
use std::ops::{BitAnd, BitOr, BitXor, Not};
use std::ptr::null_mut;
use std::simd::Simd;
use std::slice::from_raw_parts_mut;
use std::{alloc::Layout, fmt::Debug};
use std::{println, todo};

//-----------------STORAGE-----------------//
type SparseIndex = usize;
type DenseIndex = usize;
type TypeId = u64;

const MASK_HEAD: usize = 0b10000000_00000000_00000000_00000000_00000000_00000000_00000000_00000000_;
const MASK_TAIL: usize = 0b01111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111_;

trait Tuple: Sized + 'static {
    fn insert(
        self,
        table: &mut Table,
        sparse_index: Option<SparseIndex>,
    ) -> Result<(), &'static str>;
    fn remove(table: &mut Table, sparse_index: SparseIndex) -> Result<Self, &'static str>;
}

impl<C0> Tuple for (C0,)
where
    C0: 'static + Clone + Sized,
{
    fn insert(
        self,
        table: &mut Table,
        sparse_index: Option<SparseIndex>,
    ) -> Result<(), &'static str> {
        table.insert(sparse_index, self.0)?;
        Ok(())
    }
    fn remove(table: &mut Table, sparse_index: SparseIndex) -> Result<Self, &'static str> {
        Ok((table.remove::<C0>(sparse_index)?,))
    }
}
impl<C0, C1> Tuple for (C0, C1)
where
    C0: 'static + Clone + Sized,
    C1: 'static + Clone + Sized,
{
    fn insert(
        self,
        table: &mut Table,
        sparse_index: Option<SparseIndex>,
    ) -> Result<(), &'static str> {
        table.insert(sparse_index, self.0)?;
        table.insert(sparse_index, self.1)?;
        Ok(())
    }
    fn remove(table: &mut Table, sparse_index: SparseIndex) -> Result<Self, &'static str> {
        Ok((
            table.remove::<C0>(sparse_index)?,
            table.remove::<C1>(sparse_index)?,
        ))
    }
}
impl<C0, C1, C2> Tuple for (C0, C1, C2)
where
    C0: 'static + Clone + Sized,
    C1: 'static + Clone + Sized,
    C2: 'static + Clone + Sized,
{
    fn insert(
        self,
        table: &mut Table,
        sparse_index: Option<SparseIndex>,
    ) -> Result<(), &'static str> {
        table.insert(sparse_index, self.0)?;
        table.insert(sparse_index, self.1)?;
        table.insert(sparse_index, self.2)?;
        Ok(())
    }
    fn remove(table: &mut Table, sparse_index: SparseIndex) -> Result<Self, &'static str> {
        Ok((
            table.remove::<C0>(sparse_index)?,
            table.remove::<C1>(sparse_index)?,
            table.remove::<C2>(sparse_index)?,
        ))
    }
}
impl<C0, C1, C2, C3> Tuple for (C0, C1, C2, C3)
where
    C0: 'static + Clone + Sized,
    C1: 'static + Clone + Sized,
    C2: 'static + Clone + Sized,
    C3: 'static + Clone + Sized,
{
    fn insert(
        self,
        table: &mut Table,
        sparse_index: Option<SparseIndex>,
    ) -> Result<(), &'static str> {
        table.insert(sparse_index, self.0)?;
        table.insert(sparse_index, self.1)?;
        table.insert(sparse_index, self.2)?;
        table.insert(sparse_index, self.3)?;
        Ok(())
    }
    fn remove(table: &mut Table, sparse_index: SparseIndex) -> Result<Self, &'static str> {
        Ok((
            table.remove::<C0>(sparse_index)?,
            table.remove::<C1>(sparse_index)?,
            table.remove::<C2>(sparse_index)?,
            table.remove::<C3>(sparse_index)?,
        ))
    }
}

struct TypeErasedVec {
    ptr: *mut u8,
    len: usize,
    cap: usize,
    layout: Layout,
    id: TypeId,
}
impl TypeErasedVec {
    fn new_empty<C: 'static + Clone + Sized>(cap: usize) -> Self {
        assert!(cap != 0, "zero capacity is not allowed");
        let layout = Layout::new::<C>();
        assert!(layout.size() != 0, "zst");
        let repeated_layout = Layout::array::<C>(cap).unwrap();
        // let (repeated_layout, offset) = layout.repeat(cap).unwrap();
        Self {
            ptr: unsafe { alloc(repeated_layout) },
            len: 0,
            cap,
            layout,
            id: type_id::<C>(),
        }
    }

    fn push<C: 'static + Sized>(&mut self, value: C) {
        assert!(type_id::<C>() == self.id);
        self.ensure_cap(self.len + 1);
        self.len += 1;
        unsafe {
            // apparently add also adds padding for you
            *(self.ptr as *mut C).add(self.len - 1) = value;
        }
    }

    // won't do anything if it's already full
    fn populate<C: 'static + Clone + Sized>(&mut self, value: C) {
        assert!(type_id::<C>() == self.id);
        if self.len < self.cap {
            let old_len = self.len;
            self.len = self.cap;
            for each in old_len..self.len {
                unsafe {
                    *(self.ptr as *mut C).add(each) = value.clone();
                }
            }
        }
    }

    fn ensure_cap(&mut self, cap: usize) {
        if cap > self.cap {
            let rounded_num = get_num(cap);
            self.ptr = unsafe {
                realloc(
                    self.ptr,
                    self.layout.repeat(self.len).unwrap().0,
                    self.cap * rounded_num * self.layout.size(),
                )
            };
            self.cap = rounded_num;
        }
    }

    fn as_slice<'a, 'b, C: 'static + Sized>(&'a self) -> &'b mut [C] {
        assert!(type_id::<C>() == self.id);
        unsafe { from_raw_parts_mut(self.ptr as *mut C, self.len) }
    }
}
impl Drop for TypeErasedVec {
    fn drop(&mut self) {
        unsafe {
            // man i don't fucking trust my code one bit at this point, does this actually clean up the allocation for good???
            dealloc(self.ptr, self.layout.repeat(self.cap).unwrap().0);
        }
    }
}

struct DenseVec {
    sparse_index_vec: TypeErasedVec,
    comp_vec: TypeErasedVec,
}

impl Debug for DenseVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DenseVec")
            .field("sparse_index_vec", &self.as_slice())
            // .field("comp_vec", &self.comp_vec)
            .finish()
    }
}

impl DenseVec {
    fn new_empty<C: 'static + Clone + Sized>() -> Self {
        let result = Self {
            sparse_index_vec: TypeErasedVec::new_empty::<SparseIndex>(64),
            comp_vec: TypeErasedVec::new_empty::<C>(64),
        };
        result
    }

    fn push<C: 'static + Clone + Sized>(
        &mut self,
        content: C,
        sparse_index: SparseIndex,
    ) -> DenseIndex {
        self.comp_vec.push(content);
        self.sparse_index_vec.push(sparse_index);
        self.comp_vec.len - 1
    }

    fn swap_remove_non_last<C: 'static + Clone + Sized>(
        &mut self,
        mut dense_index: DenseIndex,
    ) -> C {
        let value = self.comp_vec.as_slice::<C>()[dense_index].clone();
        self.as_slice()[dense_index] = *self.as_slice().last().unwrap();
        self.comp_vec.as_slice::<C>()[dense_index] =
            self.comp_vec.as_slice::<C>().last().unwrap().clone();

        self.comp_vec.len -= 1;
        self.sparse_index_vec.len -= 1;
        value
    }

    fn pop<C: 'static + Clone + Sized>(&mut self) -> C {
        let val = self.comp_vec.as_slice::<C>().last().unwrap().clone();
        self.comp_vec.len -= 1;
        self.sparse_index_vec.len -= 1;
        val
    }

    fn as_slice(&self) -> &mut [SparseIndex] {
        self.sparse_index_vec.as_slice()
    }
}

struct SparseSet {
    sparse_vec: TypeErasedVec,
    dense_vec: DenseVec,
}
impl SparseSet {
    // this size is to make sure that later initialized column automatically resizes to this size
    fn new<C: 'static + Clone + Sized>(size: usize) -> Self {
        let mut thing = Self {
            sparse_vec: TypeErasedVec::new_empty::<DenseIndex>(size),
            dense_vec: DenseVec::new_empty::<C>(),
        };
        thing.sparse_vec.populate::<DenseIndex>(0);
        thing
    }

    fn write<C: 'static + Clone + Sized>(
        &mut self,
        sparse_index: SparseIndex,
        value: C,
    ) -> Result<(), &'static str> {
        match self.as_slice()[sparse_index] {
            0 => {
                let dense = self.dense_vec.push::<C>(value, sparse_index);
                self.as_slice()[sparse_index] = dense | MASK_HEAD;
                Ok(())
            }
            num => Err("sprase index taken"),
        }
    }

    fn clear<C: 'static + Clone + Sized>(
        &mut self,
        sparse_index: SparseIndex,
    ) -> Result<C, &'static str> {
        let mut dense_index = self.as_slice()[sparse_index];
        if dense_index != 0 {
            dense_index &= MASK_TAIL;
            self.as_slice()[sparse_index] = 0;
            let val = if dense_index == self.dense_vec.sparse_index_vec.len - 1 {
                self.dense_vec.pop()
            } else {
                let removed_val = self.dense_vec.swap_remove_non_last::<C>(dense_index);
                self.sparse_vec.as_slice()[self.dense_vec.as_slice()[dense_index]] =
                    dense_index | MASK_HEAD;
                removed_val
            };
            Ok(val)
        } else {
            Err("invalid sparse index")
        }
    }

    fn ensure_cap(&mut self, cap: usize) {
        self.sparse_vec.ensure_cap(cap);
        self.sparse_vec.populate::<usize>(0);
    }

    fn as_slice<'a, 'b>(&'a self) -> &'b mut [DenseIndex] {
        self.sparse_vec.as_slice()
    }

    fn as_simd<'a, 'b>(&'a self) -> &'b [Simd<usize, 64>] {
        assert!(self.sparse_vec.len % 64 == 0);
        unsafe {
            from_raw_parts_mut(
                self.sparse_vec.ptr as *mut Simd<usize, 64>,
                self.sparse_vec.len / 64,
            )
        }
    }
}

struct Table {
    table: HashMap<TypeId, SparseSet>,
    // sparse index cache
    freehead: SparseIndex,
    available_indices: Vec<SparseIndex>,
    num_of_components_column: TypeErasedVec,
    // buffer pool
    buffer_pool: BufferPool,
}

impl Table {
    fn new() -> Self {
        Self {
            table: HashMap::new(),
            freehead: 1,
            available_indices: vec![],
            num_of_components_column: TypeErasedVec::new_empty::<usize>(64),
            buffer_pool: BufferPool::new(8),
        }
    }

    fn ensure_cap(&mut self, cap: usize) {
        if self.num_of_components_column.cap >= cap {
            return;
        }
        self.num_of_components_column.ensure_cap(cap);
        self.num_of_components_column.populate::<usize>(0);
        for (_, v) in &mut self.table {
            v.ensure_cap(cap);
        }
    }

    fn insert<C: 'static + Clone + Sized>(
        &mut self,
        sparse_index: Option<SparseIndex>,
        value: C,
    ) -> Result<(), &'static str> {
        let sparse_index = if let Some(val) = sparse_index {
            if val >= self.freehead {
                return Err("attempting to write at an uninit index");
            }
            val
        } else {
            let sparse = if !self.available_indices.is_empty() {
                let index = *self.available_indices.last().unwrap();
                self.available_indices.pop();
                index
            } else {
                self.ensure_cap(self.freehead + 1);
                self.num_of_components_column.populate::<usize>(0);
                self.freehead += 1;
                self.freehead - 1
            };
            sparse
        };
        self.table
            .entry(std::intrinsics::type_id::<C>())
            .or_insert(SparseSet::new::<C>(self.num_of_components_column.len))
            .write(sparse_index, value)?;
        self.num_of_components_column.as_slice::<usize>()[sparse_index] += 1;
        Ok(())
    }

    fn remove<C: 'static + Clone + Sized>(
        &mut self,
        sparse_index: SparseIndex,
    ) -> Result<C, &'static str> {
        let res = self
            .table
            .get_mut(&std::intrinsics::type_id::<C>())
            .ok_or("comp type not in table")?
            .clear(sparse_index)?;
        let num = &mut self.num_of_components_column.as_slice::<usize>()[sparse_index];
        *num -= 1;
        if *num <= 0 {
            if *num == self.freehead - 1 {
                self.freehead -= 1;
            } else {
                self.available_indices.push(sparse_index);
                self.available_indices.sort_by(|x, y| y.cmp(x));
            }
        }
        Ok(res)
    }

    // it is guaranteed that all sparse vecs are of the same size
    fn get_sparse<'a, 'b>(&'a self, id: TypeId) -> Result<&'b [Simd<usize, 64>], &'static str> {
        Ok(self
            .table
            .get(&id)
            .ok_or("no such type in the system")?
            .as_simd())
    }

    fn get_num_column<'a, 'b>(&'a self) -> &'b [usize] {
        self.num_of_components_column.as_slice::<usize>()
    }

    fn get_dense<'a, 'b, C: 'static + Clone + Sized>(
        &'a self,
    ) -> Result<&'b mut [C], &'static str> {
        Ok(self
            .table
            .get(&std::intrinsics::type_id::<C>())
            .ok_or("type not in table")?
            .dense_vec
            .comp_vec
            .as_slice::<C>())
    }

    fn load_single_sparse_cache(
        &mut self,
        id: TypeId,
        invert_flag: bool,
    ) -> Result<BufferIndex, &'static str> {
        let cache = self.get_sparse(id)?;
        let buffer_id = self.buffer_pool.load(cache, invert_flag);
        Ok(buffer_id)
    }

    fn apply_operation_then_cache(
        &mut self,
        left: BufferIndex,
        right: BufferIndex,
        invert_flag: bool,
        op: Operation,
    ) -> Result<BufferIndex, &'static str> {
        let left_cache = self.buffer_pool.get(left);
        let right_cache = self.buffer_pool.get(right);
        let (buffer, buffer_index) = self.buffer_pool.request(left_cache.len());
        match op {
            Operation::And => {
                for index in 0..left_cache.len() {
                    // we actually only care about the first bit here since it is the flag
                    buffer[index] = left_cache[index] & right_cache[index];
                }
            }
            Operation::Or => {
                for index in 0..left_cache.len() {
                    buffer[index] = left_cache[index] | right_cache[index];
                }
            }
            Operation::Xor => {
                for index in 0..left_cache.len() {
                    buffer[index] = left_cache[index] ^ right_cache[index];
                }
            }
        }
        if invert_flag {
            self.buffer_pool.negate(buffer_index);
        }
        Ok(buffer_index)
    }

    fn filter_traverse(
        &mut self,
        node_index: usize,
        filter: &Filter,
    ) -> Result<BufferIndex, &'static str> {
        match &filter.nodes[node_index].unwrap() {
            Node::Single(invert_flag, id_index) => {
                self.load_single_sparse_cache(filter.ids[*id_index as usize], *invert_flag)
            }
            Node::Dual(invert_flag, node_index_left, op, node_index_right) => {
                let thing1 = self.filter_traverse(*node_index_left as usize, filter)?;
                let thing2 = self.filter_traverse(*node_index_right as usize, filter)?;
                self.apply_operation_then_cache(thing1, thing2, *invert_flag, *op)
            }
        }
    }

    fn query<'a, 'b, 'c, C: 'static + Clone + Sized>(
        &'a mut self,
        filter: &'a Filter,
    ) -> Result<(&'b mut [C], &'c [usize]), &'static str> {
        let data = self.get_dense::<C>()?;

        let bitset_index = self.filter_traverse(0, filter)?;
        let filtered_buffer = self.buffer_pool.get(bitset_index);
        let target_sparse_vec = self.get_sparse(type_id::<C>())?;

        let shift = Simd::from_array([63usize; 64]);
        let one = Simd::from_array([1usize; 64]);

        for index in 0..filtered_buffer.len() {
            filtered_buffer[index] =
                !((filtered_buffer[index] >> shift) - one) & target_sparse_vec[index];
        }

        let result_slice = self.buffer_pool.get_slice(bitset_index);

        result_slice.sort_unstable_by(|a, b| b.cmp(a));
        Ok((data, result_slice))
    }
}

// get a num that is larger or equal to num that is the 2 to the power of n
fn get_num(mut x: usize) -> usize {
    if x == 0 {
        return 0;
    }
    x -= 1;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

// essentially a bitset
struct BufferUnit {
    vec: TypeErasedVec,
    is_vacant: bool,
}
impl BufferUnit {
    fn new(size: usize) -> Self {
        let mut result = Self {
            // round up
            vec: TypeErasedVec::new_empty::<Simd<usize, 64>>(get_num(size)),
            is_vacant: true,
        };
        result.fit_for_size(size);
        result
    }

    // assuming that the cap is all dealt with
    fn fit_for_size(&mut self, len: usize) {
        if self.vec.len < len {
            for each in self.vec.len..len {
                unsafe {
                    *(self.vec.ptr as *mut Simd<usize, 64>).add(each) =
                        Simd::from_array([0usize; 64]);
                }
            }
        }
        // either way it should be resized
        self.vec.len = len;
    }

    fn as_slice<'a, 'b>(&'a self) -> &'b mut [DenseIndex] {
        unsafe { from_raw_parts_mut(self.vec.ptr as *mut usize, self.vec.len * 64) }
    }

    fn as_simd<'a, 'b>(&'a self) -> &'b mut [Simd<usize, 64>] {
        self.vec.as_slice::<Simd<usize, 64>>()
    }

    fn load(&mut self, cache: &[Simd<usize, 64>], flag: bool) {
        self.prepare(cache.len());
        self.as_simd().clone_from_slice(cache);
        if flag {
            self.negate();
        }
    }

    fn negate(&mut self) {
        for each in self.as_simd() {
            *each = !*each;
        }
    }

    fn free(&mut self) {
        for each in self.as_simd() {
            *each = Simd::from_array([0usize; 64])
        }
        self.is_vacant = true;
    }

    fn prepare(&mut self, size: usize) {
        self.vec.ensure_cap(size);
        self.fit_for_size(size);
        self.is_vacant = false;
    }
}

type BufferIndex = usize;
struct BufferPool {
    buffers: Vec<BufferUnit>,
}
impl BufferPool {
    fn new(number_of_units: usize) -> Self {
        let mut thing = Self { buffers: vec![] };
        for each in 0..number_of_units {
            thing.buffers.push(BufferUnit::new(2));
        }
        thing
    }

    fn get<'a, 'b>(&'a self, index: BufferIndex) -> &'b mut [Simd<usize, 64>] {
        if let Some(thing) = self.buffers.get(index) {
            if !thing.is_vacant {
                thing.as_simd()
            } else {
                panic!("index is vacant")
            }
        } else {
            panic!("invalid index")
        }
    }

    fn get_slice<'a, 'b>(&'a self, index: BufferIndex) -> &'b mut [usize] {
        if let Some(thing) = self.buffers.get(index) {
            if !thing.is_vacant {
                thing.as_slice()
            } else {
                panic!("index is vacant")
            }
        } else {
            panic!("invalid index")
        }
    }

    fn request<'a, 'b>(&'a mut self, size: usize) -> (&'b mut [Simd<usize, 64>], BufferIndex) {
        for (index, each) in self.buffers.iter_mut().enumerate() {
            if each.is_vacant {
                each.prepare(size);
                return (each.as_simd(), index);
            }
        }
        self.buffers.push(BufferUnit::new(size));
        (self.buffers.last().unwrap().as_simd(), self.buffers.len())
    }

    fn load(&mut self, cache: &[Simd<usize, 64>], invert_flag: bool) -> BufferIndex {
        for (index, each) in self.buffers.iter_mut().enumerate() {
            if each.is_vacant {
                each.load(cache, invert_flag);
                return index;
            }
        }
        self.buffers.push(BufferUnit::new(cache.len()));
        self.buffers.last_mut().unwrap().load(cache, invert_flag);
        self.buffers.len()
    }

    fn free(&mut self, index: BufferIndex) {
        if let Some(thing) = self.buffers.get_mut(index) {
            if thing.is_vacant {
                panic!("already vacant")
            } else {
                thing.free()
            }
        } else {
            panic!("invalid index")
        }
    }

    fn free_all(&mut self) {
        for each in self.buffers.iter_mut() {
            if !each.is_vacant {
                each.free()
            }
        }
    }

    fn negate(&mut self, index: BufferIndex) {
        if let Some(thing) = self.buffers.get_mut(index) {
            if thing.is_vacant {
                panic!("can't negate vacant buffer")
            } else {
                thing.negate();
            }
        } else {
            panic!("invalid index")
        }
    }
}

struct IterMut<'a, C: Clone + 'static + Sized> {
    data: *mut C,
    data_len: usize,
    ptr: *const DenseIndex,
    end: *const DenseIndex,
    phantom: PhantomData<&'a C>,
}
impl<'a, C: Clone + 'static + Sized> IterMut<'a, C> {
    fn new(data: &mut [C], dense_indices: &[DenseIndex]) -> Self {
        Self {
            data: data.as_mut_ptr(),
            ptr: dense_indices.as_ptr_range().start,
            end: dense_indices.as_ptr_range().end,
            phantom: PhantomData,
            data_len: data.len(),
        }
    }

    fn new_empty() -> Self {
        Self {
            data: null_mut::<C>(),
            ptr: null_mut::<DenseIndex>(),
            end: null_mut::<DenseIndex>(),
            phantom: PhantomData,
            data_len: 0,
        }
    }

    fn new_null_filter(data: &mut [C]) -> Self {
        todo!()
    }
}
impl<'a, C: Clone + 'static + Sized> Iterator for IterMut<'a, C> {
    type Item = &'a mut C;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            if self.ptr != self.end && *self.ptr != 0 {
                if *self.ptr & MASK_TAIL >= self.data_len {
                    panic!("outta bound for data vec")
                }
                let stuff = self.data.add(*self.ptr & MASK_TAIL).as_mut().unwrap();
                self.ptr = self.ptr.add(1);
                Some(stuff)
            } else {
                None
            }
        }
    }
}

// todo: caching idea, you can treat a row cache as a bitset indexed by a type id;
// you can run that row cache thru the filter and yield a boolean value, which would determine if that row fits that filter;
// this way you can cache all filters within the table and also cache all the rows and update them and run the updated row
// thru the cached filters to see if they fit (remember to eliminate repeated filters by only hashing the computed id)
// but a problem is that you need to loop thru all the cached filters all the time, which is very expensive

//-----------------FILTER-----------------//
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Operation {
    And,
    Or,
    Xor,
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Node {
    // index in ids
    Single(bool, u8),
    Dual(bool, u8, Operation, u8),
}

impl Debug for Filter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Filter")
            .field("num_of_nodes", &self.num_of_nodes)
            .field("num_of_ids", &self.num_of_ids)
            .field("nodes", &self.nodes)
            // .field("ids", &self.ids)
            .finish()
    }
}

const NUM_OF_NODES: usize = 16;
#[derive(Clone, PartialEq, Eq)]
struct Filter {
    num_of_nodes: u8,
    num_of_ids: u8,

    nodes: [Option<Node>; NUM_OF_NODES],
    ids: [TypeId; NUM_OF_NODES],
}
impl Filter {
    const fn from<C: Clone + 'static>() -> Self {
        let mut thing = Self {
            num_of_nodes: 1,
            num_of_ids: 1,
            nodes: [None; NUM_OF_NODES],
            ids: [0; NUM_OF_NODES],
        };
        thing.ids[0] = std::intrinsics::type_id::<C>();
        // first node would be the root node
        thing.nodes[0] = Some(Node::Single(false, 0));
        thing
    }

    fn combine(mut first: Self, mut second: Self, op: Operation) -> Self {
        assert!(first.nodes[0].is_some() == true);
        assert!(second.nodes[0].is_some() == true);

        first.nodes.copy_within(0..first.num_of_nodes as usize, 1);
        first.nodes[0] = None;
        for each in &mut first.nodes[1..(1 + first.num_of_nodes) as usize] {
            match each {
                Some(val) => match val {
                    Node::Single(_, _) => (),
                    Node::Dual(_, index_left, _, index_right) => {
                        match index_left {
                            finally => *finally += 1,
                        }
                        match index_right {
                            finally => *finally += 1,
                        }
                    }
                },
                None => panic!("should not be none"),
            }
        }
        first.num_of_nodes += 1;

        first.nodes
            [first.num_of_nodes as usize..(first.num_of_nodes + second.num_of_nodes) as usize]
            .clone_from_slice(&second.nodes[0..second.num_of_nodes as usize]);
        for each in &mut first.nodes
            [first.num_of_nodes as usize..(first.num_of_nodes + second.num_of_nodes) as usize]
        {
            match each {
                Some(val) => match val {
                    Node::Single(_, _) => (),
                    Node::Dual(_, index_left, _, index_right) => {
                        match index_left {
                            finally => *finally += first.num_of_nodes,
                        }
                        match index_right {
                            finally => *finally += first.num_of_nodes,
                        }
                    }
                },
                None => panic!("should not be none"),
            }
        }
        first.nodes[0] = Some(Node::Dual(false, 1, op, first.num_of_nodes));
        first.num_of_nodes += second.num_of_nodes;

        first.ids
            [first.num_of_ids as usize..first.num_of_ids as usize + second.num_of_ids as usize]
            .clone_from_slice(&second.ids[0..second.num_of_ids as usize]);
        for each in &mut first.nodes
            [(first.num_of_nodes - second.num_of_nodes) as usize..first.num_of_nodes as usize]
        {
            match each {
                Some(val) => match val {
                    Node::Single(_, index) => *index += first.num_of_ids,
                    Node::Dual(_, _, _, _) => (),
                },
                None => panic!("should not be none"),
            }
        }
        first.num_of_ids += second.num_of_ids;

        first
    }

    const NULL: Filter = Self {
        num_of_nodes: 0,
        num_of_ids: 0,
        nodes: [None; NUM_OF_NODES],
        ids: [0; NUM_OF_NODES],
    };
}
impl BitAnd for Filter {
    type Output = Filter;
    fn bitand(mut self, mut rhs: Self) -> Self::Output {
        Filter::combine(self, rhs, Operation::And)
    }
}
impl BitOr for Filter {
    type Output = Filter;
    fn bitor(mut self, mut rhs: Self) -> Self::Output {
        Filter::combine(self, rhs, Operation::Or)
    }
}
impl BitXor for Filter {
    type Output = Filter;
    fn bitxor(mut self, mut rhs: Self) -> Self::Output {
        Filter::combine(self, rhs, Operation::Xor)
    }
}
impl Not for Filter {
    type Output = Filter;
    fn not(mut self) -> Self::Output {
        match &mut self.nodes[0] {
            Some(val) => match val {
                Node::Single(flag, _) => *flag = !*flag,
                Node::Dual(flag, _, _, _) => *flag = !*flag,
            },
            None => panic!("invalid filter"),
        }
        self
    }
}
//-----------------COMMAND-----------------//
struct Command<'a> {
    table: &'a mut Table,
    system: &'a mut System,
}
impl<'a> Command<'a> {
    fn new(table: &'a mut Table, system: &'a mut System) -> Self {
        Self { table, system }
    }

    // how am i supposed to get that sparese index to even get that sparse index tho
    fn write<T: Tuple>(&mut self, comps: T, sparse_index: Option<SparseIndex>) {
        comps.insert(self.table, sparse_index).unwrap();
    }

    fn remove<T: Tuple>(&mut self, sparse_index: SparseIndex) -> Result<T, &'static str> {
        T::remove(self.table, sparse_index)
    }

    fn query<C: 'static + Clone + Sized>(&mut self) -> IterMut<C> {
        if self.system.filter == Filter::NULL {
            return IterMut::new_null_filter();
        }
        match self.table.query::<C>(&self.system.filter) {
            Ok(val) => IterMut::new(val.0, val.1),
            Err(_) => IterMut::new_empty(),
        }
    }
}

//-----------------SCHEDULER-----------------//
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
enum ExecutionFrequency {
    Always,
    Once,
    // Timed(f64, f64),
}

struct System {
    order: usize,
    frequency: ExecutionFrequency,
    func: fn(Command),
    run_times: usize,
    filter: Filter,
}
impl System {
    fn run(&mut self, table: &mut Table) {
        (self.func)(Command::new(table, self));
        self.run_times += 1;
    }
}

impl PartialEq for System {
    fn eq(&self, other: &Self) -> bool {
        self.order == other.order
    }
}
impl Eq for System {}
impl PartialOrd for System {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.order.partial_cmp(&other.order)
    }
}
impl Ord for System {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.order.cmp(&other.order)
    }
}

struct Scheduler {
    new_pool: Vec<System>,
    queue: Vec<System>,
}
impl Scheduler {
    fn new() -> Self {
        Self {
            new_pool: vec![],
            queue: vec![],
        }
    }

    fn prepare_queue(&mut self) {
        self.queue.retain(|x| match x.frequency {
            ExecutionFrequency::Always => return true,
            ExecutionFrequency::Once => {
                if x.run_times > 0 {
                    return false;
                } else {
                    return true;
                }
            }
        });
        if !self.new_pool.is_empty() {
            self.queue.append(&mut self.new_pool);
            self.new_pool.clear();
        }
        self.queue.sort();
    }
}

struct ECS {
    table: Table,
    scheduler: Scheduler,
}

impl ECS {
    fn new() -> Self {
        Self {
            table: Table::new(),
            scheduler: Scheduler::new(),
        }
    }

    fn add_system(&mut self, func: fn(Command), order: usize, once: bool, filter: Filter) {
        let frequency = match once {
            true => ExecutionFrequency::Once,
            false => ExecutionFrequency::Always,
        };
        self.scheduler.new_pool.push(System {
            order,
            frequency,
            func,
            run_times: 0,
            filter,
        })
    }

    fn tick(&mut self) {
        self.table.buffer_pool.free_all();
        self.scheduler.prepare_queue();
        for system in &mut self.scheduler.queue {
            system.run(&mut self.table);
        }
    }
}

#[cfg(test)]
mod test {
    use std::println;

    use super::*;

    #[derive(Clone, Debug)]
    struct Health(i32);
    const HEALTH: Filter = Filter::from::<Health>();

    #[derive(Clone, Debug, PartialEq)]
    struct Mana(i32);
    const MANA: Filter = Filter::from::<Mana>();

    #[derive(Clone, Debug)]
    struct Player(&'static str);
    const PLAYER: Filter = Filter::from::<Player>();

    #[derive(Clone, Debug)]
    struct Enemy(&'static str);
    const ENEMY: Filter = Filter::from::<Enemy>();

    fn look_up(id: u64) -> &'static str {
        let health = type_id::<Health>();
        let enemy = type_id::<Enemy>();
        let mana = type_id::<Mana>();
        let player = type_id::<Player>();

        if id == health {
            "health"
        } else if id == enemy {
            "enemy"
        } else if id == mana {
            "mana"
        } else if id == player {
            "player"
        } else {
            "unknown"
        }
    }

    #[derive(Clone, Debug)]
    struct Data {
        first: [u8; 8],
        bool: [bool; 63],
    }

    #[test]
    fn type_erased_vec() {
        let mut vec = TypeErasedVec::new_empty::<usize>(1);
        for each in 0..765 {
            vec.push::<usize>(1);
        }
        assert!(vec.cap == 1024);
        assert!(vec.len == 765);
        let mut reference_vec = vec![1usize; 765];
        assert!(vec.as_slice::<usize>() == &reference_vec[..]);
        vec.populate::<usize>(2);
        reference_vec.append(&mut vec![2; get_num(765) - 765]);
        assert!(vec.as_slice::<usize>() == &reference_vec[..]);
        assert!(vec.len == 1024);
        assert!(vec.cap == 1024);
    }

    #[test]
    fn dense_vec() {
        let mut dense_vec = DenseVec::new_empty::<String>();

        for each in 1..5 {
            let mut string = String::from("content ");
            string.push_str(&each.to_string());
            let dense_index = dense_vec.push::<String>(string, each);
            assert!(dense_index == each - 1)
        }
        assert!(
            dense_vec.comp_vec.as_slice::<String>()
                == &["content 1", "content 2", "content 3", "content 4"]
        );
        assert!(dense_vec.as_slice() == &[1, 2, 3, 4]);

        dense_vec.swap_remove_non_last::<String>(1);
        assert!(
            dense_vec.comp_vec.as_slice::<String>() == &["content 1", "content 4", "content 3"]
        );
        assert!(dense_vec.as_slice() == &[1, 4, 3]);

        dense_vec.push::<String>("uwu please work".into(), 666);
        assert!(
            dense_vec.comp_vec.as_slice::<String>()
                == &["content 1", "content 4", "content 3", "uwu please work"]
        );
        assert!(dense_vec.as_slice() == &[1, 4, 3, 666]);
    }

    #[test]
    fn sparse_set_insert_remove() {
        let mut sparse_set = SparseSet::new::<String>(1);
        sparse_set.ensure_cap(64);

        sparse_set.write::<String>(12, "value 12".into()).unwrap();
        assert!(
            sparse_set.as_slice()[12]
                & 0b01111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111
                == 0usize
        );
        assert!(sparse_set.dense_vec.as_slice()[0] == 12);
        assert!(sparse_set.dense_vec.comp_vec.as_slice::<String>()[0] == "value 12");

        let string = sparse_set.clear::<String>(12).unwrap();
        assert!(string == "value 12".to_string());
        assert!(sparse_set.dense_vec.as_slice() == &[]);
        assert!(sparse_set.dense_vec.comp_vec.as_slice::<String>().len() == 0);
        sparse_set.write::<String>(12, "value 12".into()).unwrap();

        // println!(
        //     "sparse: {:?}, len: {:?}",
        //     sparse_set.as_slice(),
        //     sparse_set.sparse_vec.len
        // );
        sparse_set.write::<String>(63, "value 63".into()).unwrap();
        assert!(
            sparse_set.as_slice()[63]
                & 0b01111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111
                == 1usize
        );
        assert!(sparse_set.dense_vec.as_slice()[1] == 63);
        assert!(sparse_set.dense_vec.comp_vec.as_slice::<String>()[1] == "value 63");

        assert!(sparse_set.write::<String>(63, "value 63".into()) == Err("sprase index taken"));

        let string = sparse_set.clear::<String>(12).unwrap();

        assert!(&string == &"value 12");

        sparse_set.ensure_cap(200);
        assert!(sparse_set.sparse_vec.len == 256);
        assert!(sparse_set.sparse_vec.cap == 256);
        assert!(sparse_set.dense_vec.comp_vec.len == 1);
        assert!(sparse_set.dense_vec.comp_vec.cap == 64);
        for each in 100..200 {
            sparse_set.write::<String>(each, "fluff".into()).unwrap();
        }
        sparse_set
            .write::<String>(12, "content 12 but different uwu".into())
            .unwrap();
        assert!(
            sparse_set.as_slice()[12]
                & 0b01111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111
                == 101usize
        );
        assert!(sparse_set.dense_vec.as_slice()[101] == 12);
        assert!(
            sparse_set.dense_vec.comp_vec.as_slice::<String>()[101]
                == "content 12 but different uwu"
        );

        println!(
            "sparse: {:?}",
            &sparse_set
                .as_slice()
                .iter()
                .map(|x| if *x == 0 { None } else { Some(x & MASK_TAIL) })
                .collect::<Vec<Option<usize>>>()[0..6]
        );
        println!("dense: {:?}", sparse_set.dense_vec.as_slice());
    }

    #[test]
    fn buffer_unit() {
        let mut unit = BufferUnit::new(1);
        unit.load(&unit.as_simd(), true);
        assert!(unit.as_simd() == &[Simd::from_array([!0usize; 64])]);
        assert!(unit.is_vacant == false);

        unit.load(&[Simd::from_array([0usize; 64]); 3], true);
        assert!(
            unit.as_simd()
                == &[
                    Simd::from_array([!0usize; 64]),
                    Simd::from_array([!0usize; 64]),
                    Simd::from_array([!0usize; 64])
                ]
        );
        assert!(unit.is_vacant == false);

        unit.free();
        assert!(
            unit.as_simd()
                == &[
                    Simd::from_array([0usize; 64]),
                    Simd::from_array([0usize; 64]),
                    Simd::from_array([0usize; 64])
                ]
        );
        assert!(unit.is_vacant == true);

        unit.load(&[Simd::from_array([0usize; 64])], false);
        assert!(unit.as_simd() == &[Simd::from_array([0usize; 64])]);
        assert!(unit.is_vacant != true);
    }

    #[test]
    fn buffer_pool() {
        let mut pool = BufferPool::new(0);

        pool.load(&[Simd::from_array([233usize; 64]); 2], true);
        assert!(
            pool.get(0)
                == &mut [
                    Simd::from_array([!233usize; 64]),
                    Simd::from_array([!233usize; 64])
                ]
        );
        assert!(pool.buffers[0].is_vacant == false);

        pool.load(&[Simd::from_array([78usize; 64]); 2], false);
        assert!(
            pool.get(1)
                == &mut [
                    Simd::from_array([78usize; 64]),
                    Simd::from_array([78usize; 64])
                ]
        );
        assert!(pool.buffers[1].is_vacant == false);

        pool.negate(0);
        assert!(
            pool.get(0)
                == &mut [
                    Simd::from_array([233usize; 64]),
                    Simd::from_array([233usize; 64])
                ]
        );

        pool.free_all();
        assert!(
            pool.buffers[0].as_simd()
                == &mut [
                    Simd::from_array([0usize; 64]),
                    Simd::from_array([0usize; 64])
                ]
        );
        assert!(
            pool.buffers[1].as_simd()
                == &mut [
                    Simd::from_array([0usize; 64]),
                    Simd::from_array([0usize; 64])
                ]
        );

        let (slice, index_0) = pool.request(3);
        assert!(index_0 == 0);
        assert!(
            pool.get(index_0)
                == &mut [
                    Simd::from_array([0usize; 64]),
                    Simd::from_array([0usize; 64]),
                    Simd::from_array([0usize; 64])
                ]
        );
        assert!(
            slice
                == &mut [
                    Simd::from_array([0usize; 64]),
                    Simd::from_array([0usize; 64]),
                    Simd::from_array([0usize; 64])
                ]
        );

        let (slice_another, index_1) = pool.request(5);
        assert!(index_1 == 1);
        assert!(
            pool.get(index_1)
                == &mut [
                    Simd::from_array([0usize; 64]),
                    Simd::from_array([0usize; 64]),
                    Simd::from_array([0usize; 64]),
                    Simd::from_array([0usize; 64]),
                    Simd::from_array([0usize; 64])
                ]
        );
        assert!(
            slice_another
                == &mut [
                    Simd::from_array([0usize; 64]),
                    Simd::from_array([0usize; 64]),
                    Simd::from_array([0usize; 64]),
                    Simd::from_array([0usize; 64]),
                    Simd::from_array([0usize; 64])
                ]
        );
    }

    #[test]
    fn table_insert_remove() {
        let mut table = Table::new();
        for each in 0..100 {
            table.insert::<Mana>(None, Mana(each)).unwrap();
        }
        for each in 1..=100 {
            assert!(table.remove::<Mana>(each).unwrap() == Mana((each - 1) as _));
        }
    }

    #[test]
    fn filter() {
        let filter = MANA ^ !(PLAYER & ENEMY | HEALTH);
        println!("{:?}", filter);
        println!(
            "{:?}",
            filter
                .ids
                .iter()
                .map(|x| look_up(*x))
                .collect::<Vec<&str>>()
        );
    }

    #[test]
    fn table_filter() {
        let mut table = Table::new();

        table.insert::<Mana>(None, Mana(10)).unwrap();
        table.insert::<Mana>(None, Mana(10)).unwrap();

        table.insert::<Health>(Some(2), Health(100)).unwrap();

        let filter = MANA & HEALTH;
        let final_index = table.filter_traverse(0, &filter).unwrap();

        println!(
            "{:?}",
            table
                .buffer_pool
                .get_slice(final_index)
                .iter()
                .map(|x| x >> 63)
                .collect::<Vec<usize>>()
        );
    }
    #[test]
    fn iter() {
        let mut vec_content = TypeErasedVec::new_empty::<Mana>(1);
        let mut vec_index = TypeErasedVec::new_empty::<usize>(1);
        for each in 0..11 {
            vec_content.push::<Mana>(Mana(each as _));
        }

        println!("{:?}", vec_content.as_slice::<Mana>());

        for each in 0..10 {
            vec_index.push::<usize>(each + 1);
        }

        println!("{:?}", vec_index.as_slice::<usize>());
    }

    #[test]
    fn query() {
        let mut table = Table::new();
        for each in 1..=10 {
            table.insert::<Mana>(None, Mana(each * 2)).unwrap();
        }
        table.insert::<Health>(Some(5), Health(666)).unwrap();
        table.insert::<Health>(None, Health(666777)).unwrap();
        let filter = MANA ^ HEALTH;
        let (data, slice) = table.query::<Health>(&filter).unwrap();
        let iter = IterMut::new(data, slice);

        for each in iter {
            println!("{:?}", each);
        }
    }
}
