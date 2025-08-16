# 🤝 Contributing to Pokemon LoRA Generator

Thank you for your interest in contributing to the Pokemon LoRA Generator! This document provides guidelines and information for contributors.

## 🌟 Ways to Contribute

### 🐛 Bug Reports
- Use the [GitHub Issues](https://github.com/syashu16/pokemon-lora-generator/issues) page
- Include system information (OS, GPU, Python version)
- Provide steps to reproduce the issue
- Include error messages and logs

### 💡 Feature Requests
- Describe the feature and its use case
- Explain why it would be valuable
- Consider implementation complexity
- Provide mockups or examples if applicable

### 🎨 Art Contributions
- Share your generated Pokemon creations
- Contribute to the example gallery
- Help improve prompt templates
- Create tutorial content

### 📝 Documentation
- Improve README and setup instructions
- Write tutorials and guides
- Fix typos and clarify explanations
- Translate documentation to other languages

### 🔧 Code Contributions
- Fix bugs and improve performance
- Add new features
- Improve memory optimization
- Enhance the web interface

## 🚀 Getting Started

### Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/pokemon-lora-generator.git
   cd pokemon-lora-generator
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development tools
   ```

4. **Run tests**
   ```bash
   python -m pytest tests/
   ```

### Development Tools

We use these tools for code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **pytest**: Testing

Run before submitting:
```bash
black .
isort .
flake8 .
pytest
```

## 📋 Pull Request Process

### Before Submitting

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the existing code style
   - Add tests for new features
   - Update documentation as needed

3. **Test your changes**
   ```bash
   python -m pytest tests/
   python generate.py --help
   streamlit run streamlit_app.py
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

### Commit Message Format

Use conventional commits:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test additions/changes
- `chore:` Maintenance tasks

Examples:
```
feat: add batch generation support
fix: resolve CUDA memory leak in training
docs: update installation instructions
style: format code with black
```

### Pull Request Template

When creating a PR, include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Tests pass locally
- [ ] Manual testing completed
- [ ] New tests added (if applicable)

## Screenshots
(If applicable)

## Checklist
- [ ] Code follows project style
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

## 🎯 Development Guidelines

### Code Style

- **Python**: Follow PEP 8
- **Line length**: 88 characters (Black default)
- **Imports**: Use isort for organization
- **Type hints**: Use where helpful
- **Docstrings**: Use for public functions

### File Organization

```
pokemon-lora-generator/
├── generate.py           # Main CLI script
├── streamlit_app.py      # Web interface
├── app.py               # Training script
├── utils/               # Utility modules
├── tests/               # Test files
├── examples/            # Example content
└── docs/                # Documentation
```

### Testing

- Write tests for new features
- Maintain test coverage above 80%
- Use pytest fixtures for common setup
- Test both CPU and GPU code paths (when possible)

### Documentation

- Update README for new features
- Add docstrings to public functions
- Include usage examples
- Keep documentation up-to-date

## 🐛 Bug Report Template

```markdown
**Bug Description**
Clear description of the bug

**Steps to Reproduce**
1. Step one
2. Step two
3. Step three

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**System Information**
- OS: [Windows/Linux/Mac]
- Python version: [3.8/3.9/3.10/3.11]
- GPU: [GTX 1650 Ti/RTX 3060/etc.]
- VRAM: [4GB/8GB/etc.]

**Error Messages**
```
Paste error messages here
```

**Additional Context**
Any other relevant information
```

## 💡 Feature Request Template

```markdown
**Feature Description**
Clear description of the proposed feature

**Use Case**
Why would this feature be useful?

**Proposed Implementation**
How might this be implemented?

**Alternatives Considered**
Other approaches you've considered

**Additional Context**
Mockups, examples, or other relevant information
```

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- Special thanks in documentation

## 📞 Getting Help

- **Discord**: [Join our community](https://discord.gg/pokemon-lora)
- **GitHub Discussions**: For questions and ideas
- **Email**: syashu16@example.com for private matters

## 📜 Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

**Positive behavior includes:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behavior includes:**
- Harassment, trolling, or discriminatory comments
- Publishing others' private information
- Spam or off-topic content
- Any conduct that would be inappropriate in a professional setting

### Enforcement

Project maintainers are responsible for clarifying standards and will take appropriate action in response to unacceptable behavior.

## 🙏 Thank You

Thank you for contributing to the Pokemon LoRA Generator! Your contributions help make AI art generation accessible to everyone.

---

**Questions?** Feel free to reach out through any of our communication channels. We're here to help! 🚀