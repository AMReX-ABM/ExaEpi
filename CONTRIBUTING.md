# Contributing

When contributing code to ExaEpi, please use clang-format.

The primary style is given in the `.clang-format` file in the root directory. However, due to limitations of `clang-format`, there are some addititional formatting adjustments that need to be made. These are executed from the wrapper script `utilities/custom-clang-format.py`, which takes a file on `stdin`, runs `clang-format` with the additional adjustments, and produces the formatted code on `stdout`.

The best way to use `custom-clang-format.py` is to integrate it into your editor/IDE. For example, with vscode, adding the following lines to the `.vscode/settings.json` file will automatically apply the formatting when saving the file:

```
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "ms-vscode.cpptools",
    "C_Cpp.clang_format_path": "${workspaceFolder}/utilities/custom-clang-format.py",
```




