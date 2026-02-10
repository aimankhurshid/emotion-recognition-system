# Remote Training Setup Plan

## Goal Description
The goal is to set up a clean, isolated git repository for the `emotion_recognition_system` project to facilitate safe remote training on a friend's laptop. Currently, the user's home directory is initialized as a git repository, which poses a risk of pushing unrelated personal files. We will fix this by initializing a repo specifically for the project.

## User Review Required
> [!WARNING]
> Your home directory `/Users/ayeman` is currently a git repository linked to your project remote. This is risky as it tracks all your personal files.
> **We will initialize a new git repository strictly within the `emotion_recognition_system` folder.**
> - This isolates your project.
> - We will force push to the remote to overwrite any accidental uploads of your home directory if they occurred.

## Proposed Changes

### Project Structure
- Initialize `git` in `/Users/ayeman/Downloads/minorproject_facial/emotion_recognition_system`.
- Create a `.gitignore` specific to Python/Deep Learning to keep the repo clean.
- Configure the remote `origin`.

### Remote Training Instructions
We will create a specific `REMOTE_SETUP.md` file containing detailed instructions for the friend's laptop.
#### [NEW] [REMOTE_SETUP.md](file:///Users/ayeman/Downloads/minorproject_facial/emotion_recognition_system/REMOTE_SETUP.md)
This file will contain:
- Cloning instructions.
- Environment setup (conda/pip).
- Running the training script.

## Verification Plan

### Automated Tests
- Verify `git status` shows only project files.
- Verify `git remote -v` is correct.

### Manual Verification
- User will push the code.
- User will clone on friend's laptop and verify it works.
