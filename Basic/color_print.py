def color_print(normal_part: str, color_part: str, clr: str):
    print(f"{normal_part} \033[4;31;42m {color_part} \033[0m")
    
if __name__ == '__main__':
    x = 123.
    color_print('Training Loss:', f'{x}', 'r')
    