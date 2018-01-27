# Model Module

The model module supports two computing frameworks of MPI and PS, which can be deployed in a single, yarn, and S3 cluster environment. Because each model has a different implementation interface under the two computing framework, the same model is placed in two files clearly.


| id | name | explan | ml type |
| :--: | :--: | :--: | :--: |
| 1 | lr | linear regression | supervised |
| 2 | fm | factorization machine | supervised |
| 3 | ffm | field-aware factorization machine | supervised |
| 4 | mf | matrix factorization | unsupervised |
