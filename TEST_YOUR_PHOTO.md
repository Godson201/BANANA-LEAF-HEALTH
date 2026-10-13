# Test a new leaf photo

For the HTML dashboard, start `Start_Dashboard.cmd` and open http://127.0.0.1:5050. Choose a leaf photo and click Analyze leaf. The Flask backend loads the saved classifier and subject screen; uploaded photos are not retained. Recognized banana leaves receive a category, while unrecognized subjects show Uncertain. Use the confirmation button only when you know the photo shows a banana leaf.

The model selector defaults to the original four-class model. The six-class research preview adds Fusarium/Panama and Moko bacterial wilt, with separate evaluation and explicit limitations. It does not predict bunchy top or Xanthomonas wilt. Use the source-linked disease guide and whole-plant checklist for additional farmer guidance.

Alternatively, open `Test_Your_Leaf.ipynb` in Jupyter or your IDE, select a Python kernel with the dependencies in `requirements-banana.txt`, run the seven cells, and use the upload widget. The saved classifier loads without retraining.

The model always chooses one of its four classes. It cannot identify every other plant or disease; its scores are not a confirmed diagnosis. A running notebook kernel is required for new uploads.
