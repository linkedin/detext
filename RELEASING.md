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
1. Increment `VERSION` in `setup.py`. You can checkout previous versions at https://pypi.org/manage/project/detext/releases/ and determine the new release version. 
    
    Please follow the following best practices for versioning: Breaking changes are indicated by increasing the major number (high risk), new non-breaking features increment the minor number (medium risk) and all other non-breaking changes increment the patch number (lowest risk). 

1. Upload the distributions
    ```shell script
    bash pypi_release.sh 
    ```
   Note that this should prepare and upload two packages (`detext` and `li_detext`) with the same version. `detext` is the oss package for public use. `li_detext` is for LI internal use only.

You can verify the release upload at https://pypi.org/manage/project/detext/releases/


## Add a Tag

Once the x.x.x version is released to PyPi, please add tag in the `release` section of the repo home page. The tag should have the same version name `vx.x.x` (eg. `v1.0.12`) as in the released PyPi package. 
