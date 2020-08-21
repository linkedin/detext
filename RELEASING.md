Release a New Version of DeText
=========
Steps to releasing a new version of DeText involves the following steps: 1) publishing the new version to PyPi, and 2) adding a new tag to the Github repo.
## Publishing to PyPi
NOTE: this guide is for DeText owners to publish new versions to PyPi

 Make sure you have the correct permission to release DeText packages. Only owners and maintainers can upload new releases for DeText. Check more details at https://pypi.org/project/detext/.

### Steps:
1. Install `twine` for uploading to PyPi
    ```shell script
    pip install twine
    ```
1. Upload the distributions
    ```shell script
    bash pypi_release.sh 
    ```
   Note that this should prepare and upload two packages: `x.x.x` and `x.x.xrc1`. The one with `rc1` suffix is for LI internal use.

You can verify the release upload at https://pypi.org/manage/project/detext/releases/


## Add a Tag

Once the x.x.x version is released to PyPi, please add tag in the `release` section of the repo home page. The tag should have the same version name `vx.x.x` (eg. `v1.0.12`) as in the released PyPi package. 
