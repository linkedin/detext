Release a New Version of DeText
=========
Steps to releasing a new version of DeText involves the following steps: 1) publishing the new version to PyPi, and 2) adding a new tag to the Github repo.
## Publishing to PyPi
NOTE: this guide is for DeText owners to publish new versions to PyPi

1. Make sure you have the correct permission to release DeText packages. Only owners and maintainers can upload new releases for DeText. Check more details at https://pypi.org/project/detext/.

2. Create a source distribution by:
* update the version in `setup.py`
* comment the following lines (for internal use, we cannot have these dependencies in the pypi package)
```
        'tensorflow==1.14.0',
        'tensorflow_ranking==0.1.4',
        'gast==0.2.2'
```
Then run the following:
```
python setup.py sdist
```

3. Install `twine` for uploading to PyPi

```
pip install twine
```

4. Upload the distribution

```
twine upload dist/*
```

You can verify the release upload at https://pypi.org/manage/project/detext/releases/


## Add a Tag

Once the x.x.x version is released to PyPi, please add tag in the `release` section of the repo home page. The tag should have the same version name `vx.x.x` (eg. `v1.0.12`) as in the released PyPi package. 
