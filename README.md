# compiler-Design
Here’s a `README.md` file for your compiler project:  

```md
# Compiler Design Project

## Overview
This project is a simple **compiler** implementation written in **Python**. It includes:
- **Lexical Analysis** (Tokenization)
- **Syntax Analysis** (Parsing using an LL(1) Grammar)
- **Semantic Analysis** (Checking variable declarations, types, function signatures, and more)
- **Abstract Syntax Tree (AST)** construction

## Features
- Supports basic programming constructs like:
  - Variable declarations (`int`, `bool`, `char`)
  - Conditional statements (`if-else`)
  - Operators (`+`, `-`, `*`, `/`, `&&`, `||`, `==`, `<`, `>`, etc.)
  - Function definitions (`int main() { ... }`)
  - Print statements (`print("Hello")`)
  - Return statements (`return 0;`)
- Checks for **syntax** and **semantic** errors
- Implements **FIRST** and **FOLLOW** set computations for parsing
- Uses **AST representation** for program structure

## File Structure
```
📂 compilerdesign
 ├── Compiler.py    # Main compiler implementation
 ├── README.md      # Project documentation
```

## Installation & Usage
### **1. Clone the Repository**
```sh
git clone https://github.com/your-username/compiler-Design.git
cd compiler-Design
```

### **2. Run the Compiler**
```sh
python Compiler.py
```

### **3. Example Code (Input)**
```c
int main() {
    int a = 5;
    int b = 10;
    bool condition = a < b;
    if (condition) {
        print("Condition is true");
    } else {
        print("Condition is false");
    }
    return 0;
}
```

### **4. Output (AST Representation)**
```sh
Parsing Successful
Semantic Analysis Successful
```

## Future Improvements
- Add support for **loops** (`for`, `while`)
- Implement **code generation** for assembly or LLVM IR
- Optimize **error handling** for better debugging

## License
This project is licensed under the MIT License.

---
Developed by **[Kiana Mahdian]**
```
