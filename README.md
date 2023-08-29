SNN6mA

(I) Datasets & Best model

Original Datasets A.thaliana and D.melanogaster attained from [1,2] are placed inside the folder 'data'. The separated trainning and test datasets are placed inside 'data/data_train_test'. The SNN6mA model that has the best performance on A.thaliana is named as 'best_model_th' while 'best_model_me' stands for the best performance of SNN6mA on D.melanogaster. Both best-performance-models are placed in the folder 'models'.

(II) DataProcessing.py

This python file is used to turn the original sequences in to one-hot encoding form.

(III)MyModel.py

This python file contains code of the model used in SNN6mA.

(IV) ModelTraining.py

It is a python file which does the training process, parameter margin m mentioned in paper can be changed at line 28. A model is saved when its evaluated AUC is larger than previous models.

(V) Contact

Please feel free to contact us, if you are interested in our work or have any suggestions and questions about our research work. E-mail: 2618076Y@student.gla.ac.uk

References
[1] Y. Zhang, Y. Liu, J. Xu, X. Wang, X. Peng, J. Song, and D.-J. Yu, “Leveraging the attention mechanism to improve the identification of DNA N6-methyladenine sites,” Briefings in Bioinformatics, vol. 22, no. 6, pp. bbab351, 2021.

[2] F. Tan, T. Tian, X. Hou, X. Yu, L. Gu, F. Mafra, B. D. Gregory, Z. Wei, and H. Hakonarson, “Elucidation of DNA methylation on N 6-adenine with deep learning,” Nature Machine Intelligence, vol. 2, no. 8, pp. 466-475, 2020.
