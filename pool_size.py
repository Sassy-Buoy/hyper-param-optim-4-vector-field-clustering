def create_list_with_product(x):
    # Initialize an empty list
    result = []

    # Calculate the product of the elements
    product = 80

    # Add elements to the list until the desired product is achieved
    for i in range(1, x):
        print(i)
        result.append(product // i)
        product //= (product // i)

    # Add the remaining element to the list
    result.append(product)

    return result

# Example usage
x = 5
result_list = create_list_with_product(x)
print(result_list)