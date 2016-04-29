# matrix_factorization
1.this code is for large scale matrix factorization problem, in a 8 core 64g mem machine,it can process 6billion user item score pair in half an on hour one epoch.  
2. need gcc 4.8 support.  
3. support hdfs or local file reading  
4. only support .gz format
5. each line contains at least 3 elements, userid \t score \t item1 \t item 2 ....  combine items with same score for faster  io speed
