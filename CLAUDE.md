Keep in mind to use the latest flecs 4 syntax not flecs 3. ie use try_get which might return a pointer and ensure which returns a mutable reference

Always use Flecs systems to implement code which must run every loop, although it can be okay to use specialized data structures for spatial acceleration too
Keep systems under 200 lines and having 7 or fewer input components
Use multiple .so's with flecs modules for distinct domains to improve compile time and better structure file traversal

Ideally, reuse systems and components when it is pragmatic and can reduce the size of the codebase. This is especially relevant to UI rendering systems.