Download here dependency projects. <br>
Once downloaded in order to proceed there are two things to do:

- Runtime fix of imports: ``sys.path.insert(0,os.path.abspath("../dependencies/VATE"))`` (Set the dependency folder in
  sys)
- Static analysis fix of imports:
    - VSCODE: ```{
        "python.analysis.extraPaths": ["./dependencies"]
    }```
    - PyCharm: ```Right-click on dependencies/ → Mark Directory as → Sources Root```

### VATE

> git clone https://github.com/Ubriacopo/VATE.git

## BrainBERT
> git clone https://github.com/Ubriacopo/BrainBERT.git

- Also download the pre-trained model from Google Drive. See the BrainBERT script to automate.