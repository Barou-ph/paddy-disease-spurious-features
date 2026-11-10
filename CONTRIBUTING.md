# Contributing Guide

## 1. Purpose

This document defines the development workflow and contribution rules for the project.

The purpose is to keep the codebase organized, avoid conflicts between team members, and ensure that all changes can be reviewed and integrated safely.

---

## 2. Branch Structure

The project uses three types of branches:

### `main`

The `main` branch contains the stable version of the project.

- Do not develop directly on `main`.
- Do not push directly to `main`.
- Changes are merged into `main` only after the project has been reviewed and considered stable.

### `develop`

The `develop` branch is the main integration branch for team development.

- All completed feature branches are merged into `develop`.
- Do not develop directly on `develop`.
- Do not push directly to `develop`.

### `feature/*`

Each member must create a separate feature branch for their assigned task.

Examples:

```text
feature/lead-baseline
feature/data-robustness
feature/proposed-method
feature/gradcam
feature/ablation-study
```

---

## 3. Development Workflow

The standard workflow is:

```text
feature/*
    │
    │ Pull Request
    ▼
develop
    │
    │ Release / final review
    ▼
main
```

Each member works independently on a `feature/*` branch.

After completing the task, the member creates a Pull Request to merge the feature branch into `develop`.

---

## 4. Starting a New Task

Before starting a new task, update the local `develop` branch:

```bash
git checkout develop
git pull origin develop
```

Create a new feature branch:

```bash
git checkout -b feature/<task-name>
```

Example:

```bash
git checkout -b feature/data-robustness
```

---

## 5. Working on a Task

Work only on the feature branch assigned to the task.

Before making a commit, check the current changes:

```bash
git status
```

Review the changed files and make sure that no unnecessary files are included.

---

## 6. Commit Guidelines

Commits should be small and focused on one logical change.

Basic workflow:

```bash
git add .
git commit -m "<commit message>"
```

Use clear commit messages that describe what was changed.

Examples:

```text
feat: implement EfficientNet-B0 baseline
fix: correct validation split
docs: update dataset instructions
refactor: reorganize training pipeline
chore: update project configuration
experiment: add variety-aware evaluation
```

Avoid vague commit messages such as:

```text
update
fix
test
abc
done
```

---

## 7. Pushing Changes

Push the feature branch to GitHub:

```bash
git push -u origin feature/<task-name>
```

Example:

```bash
git push -u origin feature/data-robustness
```

Do not push directly to:

```text
main
develop
```

---

## 8. Pull Request

When a task is completed, create a Pull Request:

```text
feature/<task-name> → develop
```

The Pull Request should clearly describe:

- What was implemented
- Why the change was needed
- How the change was tested
- Any known problems or limitations

The author should make sure the code works locally before creating the Pull Request.

The team lead reviews the Pull Request before merging it into `develop`.

---

## 9. Updating an Existing Feature Branch

If `develop` has changed while you are working on your feature, update your branch before continuing.

First update `develop`:

```bash
git checkout develop
git pull origin develop
```

Then return to your feature branch:

```bash
git checkout feature/<task-name>
```

Merge the latest `develop` into your feature branch:

```bash
git merge develop
```

Resolve any conflicts locally, test the project, and then push the updated branch.

---

## 10. Files and Data

Do not commit large or generated files to the repository unless explicitly required.

Do not commit:

- Raw datasets
- Large model checkpoints
- Training outputs
- Temporary files
- IDE configuration files
- `.env` files
- Passwords
- API keys
- Other sensitive credentials

Dataset access and setup instructions should be documented instead of committing the dataset itself.

---

## 11. Shared Code and Project Structure

Do not unnecessarily modify shared code or project structure without discussing the change with the team.

If a change affects another member's task:

1. Discuss the change with the team.
2. Agree on the expected interface or structure.
3. Implement the change.
4. Explain the change in the Pull Request.

Avoid overwriting or removing another member's work without agreement.

---

## 12. Before Creating a Pull Request

Before opening a Pull Request, check the following:

- [ ] The task is complete.
- [ ] The code runs locally.
- [ ] No unnecessary files are committed.
- [ ] No secrets or credentials are included.
- [ ] The feature branch is based on the latest `develop`.
- [ ] Existing functionality has not been unnecessarily broken.
- [ ] The Pull Request description explains the changes clearly.

---

## 13. Team Development Rules

The following rules apply to all members:

1. Use a separate `feature/*` branch for each task.
2. Do not work directly on `main`.
3. Do not work directly on `develop`.
4. Do not push directly to `main`.
5. Do not push directly to `develop`.
6. Keep commits focused on one logical change.
7. Pull the latest `develop` before starting new work.
8. Test changes before creating a Pull Request.
9. Do not commit datasets or large generated files.
10. Communicate changes that may affect another member's work.
11. Merge completed features into `develop` through Pull Requests.
12. Keep `main` stable throughout development.
