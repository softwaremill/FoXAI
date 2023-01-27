# Frequently asked questions

## Instalation problems
<details>
    <summary>
        <b>While trying to use the library installed from source you encounter error: ``RuntimeError: CUDA error: no kernel image is available for execution on the device``</b>
    </summary>

This error indicates that you actually have pytorch version which does not have CUDA enabled. To solve that you should refer to [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/). 

</details>
<details>
    <summary>
        <b>While running/installing the library, you encounter error: ``ImportError: libGL.so.1: cannot open shared object file: No such file or directory``</b>
    </summary>

You should download the following libraries (often already installed in basic distribution of Unix):
```bash
apt-get install ffmpeg libsm6 libxext6  -y
```

Source: [StackOverflow](https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo)
</details>
<details>
    <summary>
        <b>While trying to install poetry you get an error: ``ModuleNotFoundError: No module named 'distutils.cmd'``</b>
    </summary>

It helps to install the following: 
```bash
apt-get install python3-distutils
```

Source: [StackOverflow](https://askubuntu.com/questions/1239829/modulenotfounderror-no-module-named-distutils-util)
</details>

<details>
    <summary>
    <b>Error while running ``poetry install``: ``Hash for torch (1.12.1) from archive``</b>
    </summary>

This error might occur when you are in the process of installing and abort the process before it finishes. What you need to do is actually remove all the caches which will run the clean install:
```bash
rm -rf ~/.cache/pypoetry
```
Helful source: [StackOverflow](https://stackoverflow.com/questions/71001968/python-poetry-install-failure-invalid-hashes)
</details>

<details>
    <summary>
    <b>Error while running ``poetry install``</b>
    </summary>

Sometimes the error while running poetry install persists. Then it might be useful to diable parallel connections in poetry and do everything in single thread. To do so you need to run the following:

```bash
poetry config installer.parallel false
```
Helful source: [StackOverflow](https://stackoverflow.com/questions/71001968/python-poetry-install-failure-invalid-hashes)
</details>
