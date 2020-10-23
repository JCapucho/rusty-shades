fn vertex vertex_main() { a() }

fn a() { b() }
fn b() { a() }

// args: --color never check
// expected stderr:
// error: Recursive function detected
//   ┌─ ../tests/recursive-functions.rsh:3:1
//   │
// 3 │ fn a() { b() }
//   │ ^^^^^^
// 4 │ fn b() { a() }
//   │ ^^^^^^   ^^^