use serde::Serialize;
use std::clone::Clone;
use std::cmp::{Eq, PartialEq};
use std::marker::{Copy, PhantomData};

#[derive(Eq, Debug, PartialEq, Hash, Serialize)]
pub struct Handle<T>(usize, PhantomData<T>);

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Self::new(self.0)
    }
}

impl<T> Copy for Handle<T> {}

impl<T> Handle<T> {
    pub(crate) fn new(index: usize) -> Self {
        Handle(index, PhantomData)
    }

    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Debug, Serialize)]
pub struct Arena<T>(Vec<T>);

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Arena<T> {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = (Handle<T>, &T)> {
        self.0.iter().enumerate().map(|(i, v)| (Handle::new(i), v))
    }

    pub fn append(&mut self, value: T) -> Handle<T> {
        let position = self.0.len();
        self.0.push(value);
        Handle::new(position)
    }

    pub fn fetch_or_append(&mut self, value: T) -> Handle<T>
    where
        T: PartialEq,
    {
        if let Some(index) = self.0.iter().position(|d| d == &value) {
            Handle::new(index)
        } else {
            self.append(value)
        }
    }
}

impl<T> PartialEq for Arena<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> Eq for Arena<T> where T: Eq {}

impl<T> std::ops::Index<Handle<T>> for Arena<T> {
    type Output = T;

    fn index(&self, handle: Handle<T>) -> &T {
        &self.0[handle.index()]
    }
}

impl<T> std::ops::IndexMut<Handle<T>> for Arena<T> {
    fn index_mut(&mut self, handle: Handle<T>) -> &mut T {
        &mut self.0[handle.index()]
    }
}
