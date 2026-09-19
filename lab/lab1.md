# Lab 1 – Git/DVC and Data Preparation

**Name:** Malak Srour
**Dataset:** Food-11

> **Note on the DVC remote:** For this lab I used Solution 1 from the lab instructions instead of DagsHub: I set the DVC remote to a local folder on my own computer (`C:\Users\lenevo\Desktop\dvc-storage`), kept outside the git repository, because of the size of the Food-11 dataset. Wherever the lab instructions mention DagsHub or the DagsHub web UI, my equivalent is this local folder.

## 1. Checking My Tools

Before starting, I checked which of the required tools were already installed on my laptop.

```
PS C:\Users\lenevo> git --version
git version 2.43.0.windows.1
PS C:\Users\lenevo> python3 --version
Python was not found; run without arguments to install from the Microsoft Store, or disable this shortcut from Settings > Apps > Advanced app settings > App execution aliases.
PS C:\Users\lenevo> pip --version
Fatal error in launcher: Unable to create process using "C:\Users\lenevo\AppData\Local\Programs\Python\Python312\python.exe" "C:\Users\lenevo\AppData\Local\Programs\Python\Python312\Scripts\pip.exe" --version: The system cannot find the file specified.
PS C:\Users\lenevo> uv --version
uv 0.12.13 (0ebbd9274 2026-09-10 x86_64-pc-windows-msvc)
PS C:\Users\lenevo> dvc --version
3.67.1
```

## 2. GitHub Repo, uv and DVC Init

I created an empty repository named `mlops-lab-1` on GitHub, then worked from a folder on my Desktop. My first attempt to clone and initialise ran into a mix-up, because the folder already existed locally from an earlier try:

```
PS C:\Users\lenevo> cd C:\Users\lenevo\Desktop
PS C:\Users\lenevo\Desktop> git clone https://github.com/Malak-Srour/mlops-lab-1.git
fatal: destination path 'mlops-lab-1' already exists and is not an empty directory.
PS C:\Users\lenevo\Desktop> cd mlops-lab-1
PS C:\Users\lenevo\Desktop\mlops-lab-1> uv init
error: Project is already initialized in `C:\Users\lenevo\Desktop\mlops-lab-1` (`pyproject.toml` file exists)
PS C:\Users\lenevo\Desktop\mlops-lab-1> dvc init
ERROR: failed to initiate DVC - '.dvc' exists. Use `-f` to force.
PS C:\Users\lenevo\Desktop\mlops-lab-1> dvc config cache.type hardlink,symlink,copy
PS C:\Users\lenevo\Desktop\mlops-lab-1> git add .dvc .dvcignore
PS C:\Users\lenevo\Desktop\mlops-lab-1> git commit -m "Initialize git and dvc"
[master (root-commit) fabd7d3] Initialize git and dvc
 3 files changed, 8 insertions(+)
 create mode 100644 .dvc/.gitignore
 create mode 100644 .dvc/config
 create mode 100644 .dvcignore
PS C:\Users\lenevo\Desktop\mlops-lab-1> git push
fatal: No configured push destination.
PS C:\Users\lenevo\Desktop\mlops-lab-1> git remote -v
PS C:\Users\lenevo\Desktop\mlops-lab-1> git branch
* master
```

The folder already had its own local git repository, which had never been connected to GitHub, and uv and dvc were already initialised in it from before. Rather than redo that work, I connected this folder to my GitHub repo and renamed the branch to `main`:

```
PS C:\Users\lenevo\Desktop\mlops-lab-1> git remote add origin https://github.com/Malak-Srour/mlops-lab-1.git
PS C:\Users\lenevo\Desktop\mlops-lab-1> git branch -M main
PS C:\Users\lenevo\Desktop\mlops-lab-1> git push -u origin main
Enumerating objects: 6, done.
Counting objects: 100% (6/6), done.
Delta compression using up to 8 threads
Compressing objects: 100% (4/4), done.
Writing objects: 100% (6/6), 522 bytes | 261.00 KiB/s, done.
Total 6 (delta 0), reused 0 (delta 0), pack-reused 0
To https://github.com/Malak-Srour/mlops-lab-1.git
 * [new branch]      main -> main
branch 'main' set up to track 'origin/main'.
```

> **Question 1:** *Observe the files created, what do you think they contain.*
>
> **Answer:** Running `uv init` creates `pyproject.toml`, `.python-version`, `README.md` and `main.py`.
>
> - `pyproject.toml` is the project's ID card: it says what the project is called, what version it is, what version of Python it needs, and later on, what libraries it depends on (like when you run `uv add pillow`).
> - `.python-version` just says "use this Python version." It makes sure that if someone else opens the project, they use the same Python version I did, instead of whatever random version they have installed.
> - `README.md` is an empty file to write notes about the project later.
> - `main.py` is a starter script with a simple "hello world" type print statement, so you can immediately test that everything runs.


> **Question 2:** *What are the created files. What do you think they are used for? And which ones should be pushed to git?*
>
> **Answer:** Running `dvc init` created a `.dvc` folder, which holds DVC's internal configuration and cache settings, and a `.dvcignore` file, which works like `.gitignore` but for DVC. Only the configuration files should go to git: `.dvc/config`, `.dvc/.gitignore` and `.dvcignore`. The `.dvc/cache` folder should never be pushed to git, since that is where DVC keeps the actual data content, and it can get very large.

## 3. Setting Up the DVC Remote (Local Folder Instead of DagsHub)

Since the Food-11 dataset is large, I chose Solution 1 from the lab instructions and used a local folder as my DVC remote instead of DagsHub. I created a folder outside the repository and pointed DVC at it as the default remote:

```
cd C:\Users\lenevo\Desktop\mlops-lab-1
dvc remote add -d localremote "C:/Users/lenevo/Desktop/dvc-storage"
git add .dvc/config
git commit -m "Configure local folder as dvc remote"
git push
```

No username, password or `dvc remote modify ... auth` step was needed, since a local folder does not require authentication.

> **Question 3:** *Where are the credentials stored? and what are the options other than --global? Should the credentials be pushed to github?*
>
> **Answer:** Since I used a local folder instead of DagsHub, no credentials were needed for my setup, so this question does not really apply to me. For reference: if I had used DagsHub, credentials set with `--global` would be stored in my DVC global config file on my own computer, not inside the project folder. The alternative to `--global` is `--local`, which saves settings in a `.dvc/config.local` file that DVC automatically keeps out of git, or no flag at all, which saves to the shared `.dvc/config` file that does get pushed to git. Credentials should never be pushed to GitHub — that is exactly why DVC keeps the secret config (`--local` or `--global`) separate from the shared config, which only holds the remote's name and location.

## 4. Adding the Data

I moved the Food-11 dataset into `data/food11_raw/training`, `data/food11_raw/evaluation` and `data/food11_raw/validation` inside the project folder (I first named the folder "food11 dataset" and had to rename it to match the required `food11_raw` name). Before tracking it, I set `dvc config cache.type hardlink,symlink,copy`, so DVC would link the files instead of copying the whole dataset a second time into its cache, which saves a lot of time with this many images. I then ran `dvc add data`, and committed and pushed the resulting `data.dvc` pointer file.

When I checked my `dvc-storage` folder afterwards, it was still empty:

![The dvc-storage folder still empty](../images-folder/01-dvc-storage-empty.png)

*Figure 1: The dvc-storage folder still empty — a sign that `dvc push` had not actually been run yet.*

I had simply forgotten to run `dvc push`. After running it, the data files appeared in the folder as expected.

> **Question 4:** *Take a look at the .gitignore file. Explain what happened.*
>
> **Answer:** After running `dvc add data`, DVC automatically added a line to `.gitignore` telling git to ignore the `data` folder completely. This makes sense because DVC now handles versioning the actual files itself, so git only needs to track the small `data.dvc` pointer file instead of the real data.

> **Question 5:** *Do you see a .dvc file? What does it contain?*
>
> **Answer:** Yes, a `data.dvc` file was created in the project's root folder.

![Contents of the data.dvc pointer file](../images-folder/02-data-dvc-content.png)

*Figure 2: Contents of the data.dvc pointer file.*

It is a small text file containing a hash (`md5: a3a457d03c51ff8b037a833440f6ad13.dir`) that represents the entire contents of the `data` folder, the total size (1,188,442,712 bytes, about 1.1 GB), the number of files tracked (16,643), and the path (`data`) that this pointer refers to. This hash is what DVC uses to know exactly which version of the data belongs to each git commit.

## 5. Checking GitHub and the Local Remote

To check everything landed correctly, I opened my repository on GitHub:

![The mlops-lab-1 repository on GitHub](../images-folder/03-github-repo.png)

*Figure 3: The mlops-lab-1 repository on GitHub after pushing the git and dvc configuration files.*

> **Question 6:** *You can check your main branch on the github web UI. Is the code there? Is the data there? Do you have any file that points to the data location. And what about dagshub web UI do you see the data?*
>
> **Answer:** Yes, the code is on GitHub — I can see `.dvc`, `.dvcignore`, `.gitignore` and `data.dvc` on the main branch, along with the project files. No, the actual image data is not there — only the pointer file `data.dvc` is pushed, which holds a hash, size and file count instead of the real files. Since I used a local remote instead of DagsHub, I checked my local `dvc-storage` folder instead of the DagsHub web UI, and after running `dvc push`, the actual data files appeared there.

## 6. Testing With a Fresh Clone

To make sure the data really is stored separately from git, I cloned the repository into a new, temporary folder and tried to pull the data:

```
cd C:\Users\lenevo\Desktop
mkdir temp-test-clone
cd temp-test-clone
git clone https://github.com/Malak-Srour/mlops-lab-1.git
cd mlops-lab-1
dvc pull
```

![A fresh clone of the repository with no data folder](../images-folder/04-fresh-clone-no-data.png)

*Figure 4: A fresh clone of the repository in a temporary folder — only the code and the data.dvc pointer are present, the data folder itself is missing.*

> **Question 7:** *In a completely new temporary folder clone your github repo. Do you see the data folder? What dvc command is needed to get the data folder?*
>
> **Answer:** No, the data folder is not present after a plain `git clone` — only the code and the `data.dvc` pointer file come down. To actually get the data folder, the `dvc pull` command is needed. It reads the pointer file and fetches the real files from the configured remote (in my case, the local `dvc-storage` folder) into the `data` folder.

## 7. Python Script to Prepare the Image Files

ResNet expects images arranged in one folder per class, so I wrote `src/food11/data.py` to copy the `food11_raw` structure into two new folders, `food11_processed` and `food11_processed_mini`, resizing every image to 128x128 and sorting it into a folder named after its category. The mini version keeps at most 100 images per category, for quick development and testing. The full script is in Appendix A, and lives in the repo at `src/food11/data.py`.

Before writing the script, I added the Pillow library, which the script needs to open and resize images:

```
uv add pillow
```

I then ran the script and checked the results:

![food11_processed/training/Bread with 994 images](../images-folder/05-processed-bread-994.png)

*Figure 5: data/food11_processed/training/Bread after running the script — 994 images, resized and sorted by category.*

![food11_processed_mini/training/Bread capped at 100 images](../images-folder/06-processed-mini-bread-100.png)

*Figure 6: data/food11_processed_mini/training/Bread — capped at 100 images, as required.*

Both folders looked correct, so I tracked the new data with dvc and pushed it, the same way as before:

![Running the script then dvc add commit push and dvc push](../images-folder/07-script-dvc-add-push.png)

*Figure 7: Running the script, then dvc add, git commit, git push and dvc push for the processed datasets.*

While doing this, I noticed my uv project files (`pyproject.toml`, `uv.lock`, `src/`) had never actually been committed to git — only the dvc configuration had been, back in Section 2:

![git status showing untracked uv project files](../images-folder/08-git-status-untracked.png)

*Figure 8: git status showing the uv project files were still untracked.*

I added and pushed them:

![Adding committing and pushing the missing project files](../images-folder/09-git-add-commit-push-fix.png)

*Figure 9: Adding, committing and pushing the missing project files.*

## 8. Switching Between Commits

Before switching commits, the data folder contained all three datasets:

![The data folder with all three datasets before switching commits](../images-folder/10-data-folder-before-switch.png)

*Figure 10: The data folder before switching commits — food11_raw, food11_processed and food11_processed_mini all present.*

I listed the commits that changed `data.dvc`:

```
C:\Users\lenevo\Desktop\mlops-lab-1>git log --oneline -- data.dvc
7d6c223 Add food11_processed and food11_processed_mini
b4718e9 Track data folder with dvc
```

Then I checked out the older commit (from before the processed datasets existed) and ran `dvc checkout`:

![Checking out the older commit and running dvc checkout](../images-folder/11-checkout-old-commit.png)

*Figure 11: Checking out the older commit (git enters "detached HEAD" state, which is expected) and running dvc checkout.*

![The data folder after checking out the older commit, only food11_raw remains](../images-folder/12-data-folder-old-commit.png)

*Figure 12: The data folder after checking out the older commit — only food11_raw remains.*

> **Question 8:** *Do you still see the new folders you created? food11_processed and food11_processed_mini?*
>
> **Answer:** No. After checking out the older commit and running `dvc checkout`, only the `food11_raw` folder was there — `food11_processed` and `food11_processed_mini` were both gone, because that commit was from before those folders existed. This shows that git and DVC keep code and data in sync: going back to an old commit also rewinds the data to match.

I then returned to the present:

```
C:\Users\lenevo\Desktop\mlops-lab-1>git checkout main
```

![Running dvc checkout again after switching back to main](../images-folder/13-checkout-main-again.png)

*Figure 13: Running dvc checkout again after switching back to the main branch.*

![The data folder restored with all three datasets back](../images-folder/14-data-folder-restored.png)

*Figure 14: The data folder restored — all three datasets are back.*
