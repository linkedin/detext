Release a New Version of DeText
=========
NOTE: this guide is for DeText owners to publish new versions to PyPi

Make sure you have the correct permission to release DeText packages. Only owners and maintainers can upload new releases for DeText. Check more details at https://pypi.org/project/detext/.


## When to release
Releasing a new version for adding new features and bug fixes. Minor updates to the text (eg. typos in README.md) shall be combined with a later release.


## How to release

### Step 1: increment the version and publish to PyPi
Releasing the package involves:
* Incrementing the version of DeText
* Publishing to pypi: Note that this prepares and uploads two packages (`detext` and `detext-nodep`) with the same version. `detext` is the oss package for public use. `detext-nodep` is for LI internal use only, without any dependencies such as `tensorflow` pulled in.

Please ensure all changes are merged before releasing. Use the following command to release a new package:
```shell script
bash pypi_release.sh <part>
```
Where `part` is a required argument. It specifies the part of the version to increase, used in `bump2version`. Valid values are: `patch`, `minor`, and `major`. See `pypi_release.sh` for more details.

#### Examples:

Assume the current version in `.bumpversion.cfg` is 0.0.1.

* 0.0.1 -> 0.0.2:
    ```shell script
     bash pypi_release.sh patch
    ```
* 0.0.1 -> 0.1.0:
    ```shell script
     bash pypi_release.sh minor
    ```
* 0.0.1 -> 1.0.0:
    ```shell script
     bash pypi_release.sh major
    ```

The `.bumpversion.cfg` is the single source of truth for versioning DeText. You do not need to manually update the version number. Both `.bumpversion.cfg` and `setup.py` will be updated automatically. More about `bump2version`: https://github.com/c4urself/bump2version.
#### Best practices for versioning 
* Breaking changes are indicated by increasing the major number (high risk)
* New non-breaking features increment the minor number (medium risk)
* All other non-breaking changes increment the patch number (lowest risk). 

### Step 2: merge version changes
Running the releasing script automatically creates a new commit that includes the version update and a new tag. 

You can verify the new releases at https://pypi.org/project/detext/ and https://pypi.org/project/detext-nodep/. If the packages are successfully published, create a PR and merge to master.


### Step 3: add a Tag

Once the x.x.x version is released to PyPi, please add tag in the `release` section of the repo home page. The tag should have the same version name `vx.x.x` (eg. `v1.0.12`) as in the released PyPi package. 
