import numpy as np

def load_images(filepath):
    with open(filepath, 'rb') as f:
        # 读取文件头部
        magic_number = int.from_bytes(f.read(4), 'big')  # 魔数
        num_images = int.from_bytes(f.read(4), 'big')   # 图片数量
        rows = int.from_bytes(f.read(4), 'big')         # 图片高度
        cols = int.from_bytes(f.read(4), 'big')         # 图片宽度
        
        # 读取图片数据
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        images = image_data.reshape(num_images, rows, cols)  # 转为图片格式
        return images

def load_labels(filepath):
    with open(filepath, 'rb') as f:
        # 读取文件头部
        magic_number = int.from_bytes(f.read(4), 'big')  # 魔数
        num_labels = int.from_bytes(f.read(4), 'big')   # 标签数量
        
        # 读取标签数据
        label_data = np.frombuffer(f.read(), dtype=np.uint8)
        return label_data

if __name__=="__main__":
    # 示例：读取训练集标签
    train_labels = load_labels('train-labels.idx1-ubyte')


    # 示例：读取训练集图片
    train_images = load_images('train-images.idx3-ubyte')/255.0

    print(train_images[0])
