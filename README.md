![Alt Text](README_data/Paper_screnshot.png)

# Contents
This folder contains the Python scripts required to reproduce the examples from [this paper](https://www.youtube.com/watch?v=dQw4w9WgXcQ).

# Example 1
## Polynomial approximation (section 4.1)
![Alt Text](README_data/Example1.png)
To reproduce this example,use the [Example.py](Example.py) with the following parameters:


```python
if __name__ == '__main__':

    number_of_functions = 6
    number_of_candidate_Gauss_points = 20

    function_to_use = 1 # 1 or 2
    constrain_sum_of_weights = False #this avoids the trivial solution
    use_L2_weighting = True # True  # if True: d = G@\sqrt{W}; elif False: d = G@W

    run_example(number_of_functions, number_of_candidate_Gauss_points, function_to_use, constrain_sum_of_weights, use_L2_weighting)
```


# Example 2
##  Set of polynomial functions plus constant function (section 4.2)
![Alt Text](README_data/Example2.png)

To reproduce this example,use the [Example.py](Example.py) with the following parameters:
```python
if __name__ == '__main__':

    number_of_functions = 20
    number_of_candidate_Gauss_points = 50

    function_to_use = 2 # 1 or 2
    constrain_sum_of_weights = False #this avoids the trivial solution
    use_L2_weighting = True # True  # if True: d = G@\sqrt{W}; elif False: d = G@W

    run_example(number_of_functions, number_of_candidate_Gauss_points, function_to_use, constrain_sum_of_weights, use_L2_weighting)
```
