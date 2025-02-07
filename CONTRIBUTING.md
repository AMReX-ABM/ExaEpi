# Contributing

These are a few short instructions for coding standards when contributing to ExaEpi.

## Automatic styling

Please use clang-format for automatically formatting code.

The primary style is given in the `.clang-format` file in the root directory. However, due to limitations of `clang-format`, there are some addititional formatting adjustments that need to be made. These are executed from the wrapper script `utilities/custom-clang-format.py`, which takes a file on `stdin`, runs `clang-format` with the additional adjustments, and produces the formatted code on `stdout`.

An example of using it would be:

```
custom-clang-format.py < foo.cpp > foo_formatted.cpp
```

The best way to use `custom-clang-format.py` is to integrate it into your editor/IDE. For example, with vscode, adding the following lines to the `.vscode/settings.json` file will automatically apply the formatting when saving the file:

```
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "ms-vscode.cpptools",
    "C_Cpp.clang_format_path": "${workspaceFolder}/utilities/custom-clang-format.py",
```

## Class and function naming

We adhere to the camelCaps naming for classes, structs and functions. For function names, the first letter is lower case, whereas for classes and structs, the first letter is capitalized.

## Variable naming

Variables are typically all lowercase with underscores.

