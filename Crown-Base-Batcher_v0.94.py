"""
Crown-Base Batcher 0.94
by teon-u (github)
update date : 2023-12-18
[Update Log] 0.9 > 0.91
 - 배치 알고리즘 오류 해결 불가. 기존 작동되는 코드를 그대로 활용할 수 있도록 조정.
 - 이를 위해, 2D PNG 전처리 결과를 지금의 온 메모리 변수형태가 아닌 temp 폴더에 저장.
[Update Log] 0.91 > 0.92
 - 코드 정리 (정상작동 점검)
 - 필수 로그 스크립트 추가 (배치 순서 등)
 - 가이드라인 메시지 추가
[Update Log] 0.92 > 0.93
 - Pyinstaller 활용한 exe파일 패키징
[Update Log] 0.93 > 0.94
 - 배치순서 오류수정

[추가 개발 가능 기능]
1. 다중 BASE 데이터 처리
2. BASE 데이터 평가 (잔여배치공간 확인 및 활용)
3. 두께 데이터 반영, 배치가능여부 판단기능 추가
4. 입력된 다중 BASE 데이터에 끼워넣을수 있는 구석 찾아서 배치하는 기능
5. 원본파일 (밀링 프로그램에서 쓰는) 정보추출 및 활용
6. 배치 알고리즘 효율성 개선 (시간단축)
"""



######################
#### 1. 라이브러리 ####
######################
# UI
import tkinter as tk
from tkinter import filedialog
from tkinter import font
import sys
import json
# Pre-process 2D PNG
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
# Pre-process 3D STL
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from io import BytesIO
from PIL import Image
# Batch Algorithm
from PIL import ImageDraw, ImageFont



#############################
#### 2. 환경설정 관련 함수 ####
##############################

def get_current_directory():
    """
    Get the current directory. Works in both Jupyter Notebook and Python script environments.
    """
    try:
        # 일반 Python 스크립트 환경
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Jupyter Notebook 환경
        return os.getcwd()



################################
#### 3. UI 관련 함수 & 클래스 ####
#################################

class TextRedirector(object): # 스크립트 표현 인터페이스에
    def __init__(self, widget):
        self.widget = widget
        self.original_stdout = sys.stdout  # 원래의 stdout 저장

    def write(self, str):
        if self.widget.winfo_exists():  # 위젯이 존재하는지 확인
            self.widget.insert(tk.END, str)
            self.widget.see(tk.END)
            self.widget.update_idletasks()  # Tkinter 윈도우 업데이트
        else:
            self.original_stdout.write(str)  # 위젯이 없으면 원래 stdout으로 출력

    def flush(self):
        pass

def validate_number(P): # 유효성 검증 : 입력값이 반드시 숫자
    return P.isdigit() or P == ""

# 창 닫을 때
def on_closing(): # 필요한 정리 작업을 수행
    print("Closing application")
    sys.stdout = sys.__stdout__  # stdout을 원래대로 복원
    root.quit()  # 이벤트 루프 중단
    root.destroy()  # 윈도우 파괴

"""
편의성 : 열었던 폴더를 각각 기억하고 다시 열어주는 기능
<<설명>>
'last_dirs.json' 파일에 마지막으로 열었던 디렉토리를 각각 저장
프로그램 재실행시에도 json 파일의 경로를 따라 디렉토리를 열어줌
<<영향>>
def save_last_opened_dir
def load_last_opened_dir
def open_a_files
def open_b_files
"""
last_dir_file = 'last_dirs.json'
# 열었던 디렉토리 저장
def save_last_opened_dir(dir_type, directory):
    try:
        if os.path.exists(last_dir_file):
            with open(last_dir_file, 'r') as file:
                last_dirs = json.load(file)
        else:
            last_dirs = {}

        last_dirs[dir_type] = directory

        with open(last_dir_file, 'w') as file:
            json.dump(last_dirs, file)
    except Exception as e:
        print(f"Error saving last opened directory: {e}")
# 열었던 디렉토리 로드
def load_last_opened_dir(dir_type):
    default_dir = '/'  # 기본 디렉토리 설정 (루트 디렉토리 또는 사용자 홈 디렉토리)

    try:
        if os.path.exists(last_dir_file):
            with open(last_dir_file, 'r') as file:
                last_dirs = json.load(file)
                last_dir = last_dirs.get(dir_type, default_dir)
                # 해당 디렉토리가 실제로 존재하는지 확인
                return last_dir if os.path.exists(last_dir) else default_dir
        else:
            return default_dir
    except Exception as e:
        #print(f"Error loading last opened directory: {e}")
        return default_dir

# A file(Base) 열기
def open_a_files(): # File Selection Fucntion
    global last_opened_dir_a
    initial_dir = load_last_opened_dir('a')
    #file_paths = filedialog.askopenfilenames(initialdir=initial_dir) # 다중 파일 선택
    #last_opened_dir_a = os.path.dirname(file_path)
    #save_last_opened_dir('a', last_opened_dir_a)
    #for file_path in file_paths:
    #    listbox_a.insert(tk.END, file_path)
    file_path = filedialog.askopenfilename(initialdir=initial_dir) # 단일 파일 선택
    if file_path:  # 파일이 선택되었는지 확인
        last_opened_dir_a = os.path.dirname(file_path)
        save_last_opened_dir('a', last_opened_dir_a)
        listbox_a.delete(0, tk.END)  # 기존에 listbox_a에 있던 내용을 모두 삭제
        listbox_a.insert(tk.END, file_path)

# B file(Base) 열기
def open_b_files():
    global last_opened_dir_b
    initial_dir = load_last_opened_dir('b')
    file_paths = filedialog.askopenfilenames(initialdir=initial_dir)
    for file_path in file_paths:
        last_opened_dir_b = os.path.dirname(file_paths[0])
        save_last_opened_dir('b', last_opened_dir_b)
        listbox_b.insert(tk.END, file_path)

def reset_a_files(): #Reset Function
    listbox_a.delete(0, tk.END)

def reset_b_files():
    listbox_b.delete(0, tk.END)



##################################
#### 4. 2D PNG 전처리 관련 함수 ####
###################################

# 색상 지정
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([140, 255, 255])

def find_largest_blue_circle(img):
    """
    주어진 이미지에서 가장 큰 파란색 원을 찾는 함수
    """
    # 전역변수 가져오기
    global lower_blue
    global upper_blue
    # 더 나은 색상 인식을 위해 HSV 형태로 변경
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 이진 마스크 생성, 파란 부분은 1로, 나머지는 0으로 계산
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
    # 이진 마스크의 윤곽선 감지
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 면적을 기준으로 가장 큰 윤곽선 탐지
    c = max(contours, key=cv2.contourArea)
    # 최소 크기로 윤곽선을 둘러싸는 원을 계산
    (x, y), r = cv2.minEnclosingCircle(c)
    # 반환값 : 이미지, 파란색 원의 중심좌표, 파란색 원의 반지름 값
    return img, (int(x), int(y)), int(r)


def standardize_image(img, center, radius, target_radius):
    """
    가장 큰 파란색 원에 맞춰 이미지 크기를 표준화하는 함수
    Parameters
        img : 원본 이미지 (BGR 형태)
        center : 파란색 원의 중심 좌표
        radius : 파란색 원의 반지름
        target_radius : 표준화할 목표 반지름
        return : 표준화된 이미지
    """
    height, width = img.shape[:2]
    # 목표 반경과 감지된 반경을 기반으로 배율 계수를 계산
    scale = target_radius / radius
    # 계산된 배율 계수를 사용해 이미지 크기 조정
    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    # 크기 조정 후 새 중심좌표 계산
    new_center = (int(center[0] * scale), int(center[1] * scale))
    # 원본과 동일한 크기의 빈 흰색 이미지 생성
    result_img = np.ones((height, width, 3), dtype=np.uint8) * 255
    # 파란색 원이 중앙에 오도록 조정된 이미지를 빈 이미지 위에 배치
    x_offset = width // 2 - new_center[0]
    y_offset = height // 2 - new_center[1]
    y1, y2 = max(0, y_offset), min(height, y_offset + scaled_img.shape[0])
    x1, x2 = max(0, x_offset), min(width, x_offset + scaled_img.shape[1])
    result_img[y1:y2, x1:x2] = scaled_img[y1-y_offset:y2-y_offset, x1-x_offset:x2-x_offset]

    # 파란색 원을 표현하는 이진 마스크 생성
    mask_circle = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask_circle, (width//2, height//2), target_radius, 255, -1)
    result_img[mask_circle == 0] = [255, 255, 255]
    return result_img


def crop_to_square(img, center, target_radius):
    """
    파란색 원을 중심으로 이미지를 정사각형으로 자르는 함수
    """
    # 이미지를 자르기 위한 좌표 설정 (원을 중심으로)
    top_left = (center[0] - target_radius, center[1] - target_radius)
    bottom_right = (center[0] + target_radius, center[1] + target_radius)
    # 파란색 원을 중심으로 이미지 자르기
    cropped_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    return cropped_img


def binary_mask_image(img):
    """
    이미지의 이진 마스크를 생성하는 기능
    """
    # 더 나은 색상 인식을 위해 이미지를 HSV 형태로 변환
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 파란색 부분을 감지하기 위한 이진 마스크를 생성
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
    # 빨간색 감지를 위한 범위 설정 (빨간색의 경우 HSV에서 0과 180에 걸쳐있기 때문에 두 부분으로 나누어 처리)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    # 빨간색 부분을 감지하기 위한 이진 마스크 생성
    mask_red1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    # 빨간색 부분의 윤곽선 감지
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 파란색 부분을 1, 나머지 부분을 0으로 하는 이진 이미지 생성
    binary_img = np.ones_like(mask_blue)
    # 빨간색 부분을 0으로 채움
    cv2.drawContours(binary_img, contours, -1, 0, -1)
    # 파란색 외의 부분을 0으로 채움
    binary_img[mask_blue == 0] = 0
    return binary_img

def process_images(image_directory):
    """
    이미지 경로 입력받고 전처리된 이미지 내뱉는 함수
    Input : image_directory(List & str)
    Output : binary_cropped_images(List & img)
    """

    # Load the images
    original_images = [cv2.imread(path) for path in image_directory]

    # Process the images to find the largest blue circle
    images_data = [find_largest_blue_circle(img) for img in original_images]

    # Find the maximum radius among all images
    # 표준화 위해 그냥 고정된 값으로 변경함
    max_radius = 267  # max([data[2] for data in images_data])

    # Standardize the images
    standardized_images = [standardize_image(data[0], data[1], data[2], max_radius) for data in images_data]

    # Crop the images to square
    cropped_images = [crop_to_square(img, (img.shape[1] // 2, img.shape[0] // 2), max_radius) for img in standardized_images]

    # Apply binary mask to the cropped images
    binary_cropped_images = [binary_mask_image(img) for img in cropped_images]

    # output_directory 설정
    output_directory = os.path.join(get_current_directory(), 'temp', '2D_PNG_PROCESSED')

    # 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    image_path_list = []
    for i, img in enumerate(binary_cropped_images):
        # 원본 이미지 경로에서 파일명 추출
        original_filename = os.path.basename(image_directory[i])
        
        # 이미지 저장 경로
        output_path = os.path.join(output_directory, original_filename)

        # 이미지 경로 저장
        image_path_list.append(output_path)

        # 파일이 이미 존재하는 경우 스킵
        if os.path.exists(output_path):
            #print(f"File {original_filename} already exists. Skipping...")
            continue

        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        
        # 이미지 저장, 주변 공백 제거
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # 현재 이미지를 저장한 후 figure 닫기

    return image_path_list


##################################
#### 5. 3D STL 전처리 관련 함수 ####
###################################

def read_stl(filename):
    """
    Read an STL file and return the vertices.
    """
    with open(filename, 'rb') as f:
        header = f.read(80)
        num_triangles = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        
        vertices = np.zeros((num_triangles, 3, 3))
        
        for i in range(num_triangles):
            f.read(12)  # skip the normal vector
            vertices[i] = np.frombuffer(f.read(36), dtype=np.float32).reshape(3, 3)
            f.read(2)  # skip the attribute byte count
            
    return vertices

def stl_top_view_and_dimensions(filename):
    """
    Generate a top view image of the STL model and return its dimensions.
    STL 파일명을 받아서 탑뷰 이미지와 dimensions 를 리턴하는 함수
    """
    # Read the STL file
    vertices = read_stl(filename)
    
    # Calculate dimensions
    width = np.max(vertices[:,:,0]) - np.min(vertices[:,:,0])
    depth = np.max(vertices[:,:,1]) - np.min(vertices[:,:,1])
    height = np.max(vertices[:,:,2]) - np.min(vertices[:,:,2])
    dimensions = (width, depth, height)

    # Calculate adjusted figsize
    base_size = 10
    figsize_x = base_size * (width / 10)
    figsize_y = base_size * (height / 10)
    
    # Plotting the 3D object
    fig = plt.figure(figsize=(figsize_x, figsize_y))
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(Poly3DCollection(vertices))
    ax.set_xlim([np.min(vertices[:,:,0]), np.max(vertices[:,:,0])])
    ax.set_ylim([np.min(vertices[:,:,1]), np.max(vertices[:,:,1])])
    ax.set_zlim([np.min(vertices[:,:,2]), np.max(vertices[:,:,2])])
    ax.axis('off')
    ax.view_init(elev=90, azim=-90)
    
    # Get the bounds of the model
    bounds = np.array([
        [np.min(vertices[:,:,0]), np.max(vertices[:,:,0])],
        [np.min(vertices[:,:,1]), np.max(vertices[:,:,1])],
        [np.min(vertices[:,:,2]), np.max(vertices[:,:,2])]
    ])

    # Set the aspect ratio for x and y axis to be equal
    ax.set_box_aspect([bounds[0][1]-bounds[0][0], bounds[1][1]-bounds[1][0], bounds[2][1]-bounds[2][0]])

    # 메모리에 이미지 저장
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    # PIL 이미지로 읽기
    pil_image = Image.open(buf)

    # PIL 이미지를 OpenCV 이미지 (NumPy 배열)로 변환
    opencv_image = np.array(pil_image)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

    plt.close(fig)

    return dimensions, opencv_image

def process_image(cv_img):
    # Convert to HSV and create mask for blue regions
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Find contours and draw them on the original image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(cv_img, contours, -1, (0, 0, 255), 2)
    
    # Convert the highlighted image to grayscale
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # Apply binary thresholding
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    return binary

def imwrite_unicode(img_path, cv_img): # cv2 UNICODE 인식 오류때문에 추가한 함수
    # OpenCV 형식의 이미지를 PIL 형식으로 변환
    pil_image = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    # PIL을 사용하여 이미지 저장
    pil_image.save(img_path)

def process_stl_files(file_paths):
    """
    Function : 3D STL 파일 전처리 진행
    Input : 원본파일 경로
    Output : 전처리된 파일 경로
    Process : Output 경로에 전처리된 파일 저장
    """
    # 현재 스크립트 위치의 temp 폴더 설정
    save_dir = os.path.join(get_current_directory(), 'temp','3D_STL_PROCESSED')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image_path_list = [] # 전처리된 이미지 경로 리스트
    for stl_file in file_paths: # 각 STL 파일 처리
        # 이미지 저장 경로
        image_name = os.path.basename(stl_file).replace('.stl', '.png')
        image_path = os.path.join(save_dir, image_name)
        image_path_list.append(image_path)
        # 이미 처리된 파일 건너뛰기
        if os.path.exists(image_path):
            #print(f"Skipping already processed file: {stl_file}")
            continue

        # STL 파일에서 상단 뷰 이미지 생성
        _, top_view_img = stl_top_view_and_dimensions(stl_file)

        # 이미지 처리
        result_img = process_image(top_view_img)

        # 결과 이미지 시각화
        #cv2.imshow("Processed Image", result_img)
        #cv2.waitKey(0)  # 사용자가 키를 누를 때까지 기다립니다.
        #cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기

        # 결과 이미지 저장
        #print(image_path)
        #save_success = cv2.imwrite(image_path, result_img) #cv2 UNICODE 인식오류로 파일명이 한글일땐 작동불가
        imwrite_unicode(image_path, result_img)
        #print(f"Processed {stl_file}")
        #print('-' * 50)
    return image_path_list



################################
### 6. 배치 알고리즘 적용 함수 ###
################################

# 이미지를 그레이스케일로 변환하고 배열로 변환하는 함수
def image_to_array(img):
    return np.array(img)

# 배열을 이미지로 변환하는 함수
def array_to_image(arr):
    return Image.fromarray(np.uint8(arr))

# 이미지를 이진화하는 함수 (흰색 영역을 1, 검은색 영역을 0으로)
def binarize_image(img):
    threshold = 128  # 임계값 설정
    img_arr = image_to_array(img)
    binary_img_arr = 1 - (img_arr > threshold)  # 흰색 부분을 1로, 검은색 부분을 0으로 변환
    return binary_img_arr

# 임플란트 이미지의 크기를 조정하는 함수
def resize_images(images, scale_factor):
    resized_images = []
    for img in images:
        original_width, original_height = img.size
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        resized_images.append(resized_img)
    return resized_images

# 이미지를 주어진 각도로 회전시키는 함수
def rotate_image(img, angle):
    return img.rotate(angle, expand=True)

# 임플란트와 숫자를 배치하는 함수
def place_implants_and_numbers(base_img, implants, rotate_angle):
    base_arr = binarize_image(base_img)
    base_height, base_width = base_arr.shape
    implant_visualization_img = Image.fromarray(np.uint8(base_arr * 255), 'L')
    placed_number_image = Image.fromarray(np.uint8(base_arr * 255), 'L')
    draw = ImageDraw.Draw(placed_number_image)

    occupied_areas = np.zeros_like(base_arr, dtype=bool)  # 배치된 임플란트 영역을 추적
    #occupied_areas[base_arr == 0] = True  # 기존에 배치할 수 없는 공간을 occupied_areas에 표시


    # 가장 작은 임플란트의 크기를 구합니다.
    #min_implant_size = implants[-1].size
    #min_implant_radius = min(min_implant_size) // 2  # 최소 크기의 반지름

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    placed_implants = []
    implant_count = len(implants)

    for idx, implant in enumerate(implants):
        placed = False

        for y in range(base_height):
            for x in range(base_width):

                # 이미 많은 임플란트가 배치된 영역을 건너뜁니다.
                if occupied_areas[y,x]:
                    continue

                # 회전 각도 부여 (외부 파라미터 적용)
                for angle in range(0, 360, rotate_angle):  # 30도 간격으로 회전 (외부 파라미터 적용)
                    rotated_implant = rotate_image(implant, angle)
                    rotated_implant_arr = binarize_image(rotated_implant)
                    r_implant_height, r_implant_width = rotated_implant_arr.shape

                    # 베이스 이미지의 경계를 넘어가지 않는지 확인
                    if y + r_implant_height <= base_height and x + r_implant_width <= base_width:
                        if np.all(base_arr[y:y+r_implant_height, x:x+r_implant_width] + rotated_implant_arr <= 1):
                            base_arr[y:y+r_implant_height, x:x+r_implant_width] += rotated_implant_arr
                            placed_implants.append((x, y, angle))
                            text_position = (x + r_implant_width // 2 - 10, y + r_implant_height // 2 - 10)
                            draw.text(text_position, str(idx + 1), fill=255, font=font)
                            placed = True
                            print("      크라운 배치 성공 : ", idx + 1,"/",implant_count)#, "Name:", implant_image_paths[idx][46:-23])
                            break
                    if placed:
                        break
                if placed:
                    break
            if placed:
                break
        if not placed:
            print("      크라운 배치 실패 : ", idx + 1,"/",implant_count)#, "Name:", implant_image_paths[idx][46:-23])

    for (x, y, angle) in placed_implants:
        rotated_implant = rotate_image(implant, angle)
        implant_visualization_img.paste(rotated_implant, (x, y), rotated_implant)

    placed_implant_image = array_to_image(base_arr)
    plt.ioff() # 대화형 모드 비활성화
    print("      크라운 배치 완료")
    return placed_implants, placed_implant_image, placed_number_image

# 이미지 크기를 가져오는 함수
def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size


### 실행 함수 추가 ### (단순화)

###############
### 실행버튼 ###
###############
def execute_action(var_desc, var_size, var_rotate):
    """
    Execute 버튼 클릭시 동작하는 코드
    """
    print("    [Crown-Base Batcher 0.94 by teon-u]")
    print("    Options")
    print("    ㄴ 배치순서 :",bool(int(var_desc))," - True 면 큰것부터, False 면 작은것부터")
    print("    ㄴ 크기배수 :",var_size," - 크라운 이미지 크기를 크기배수만큼 나눠 작게 만듦")
    print("    ㄴ 회전각도 :",var_rotate," - 회전을 시키며 크라운 배치, 360으로 설정시 회전없음")

    ### 모든 파일을 가져와서 리스트로 ###
    # 2D PNG 경로 리스트
    dir_a = [listbox_a.get(i) for i in range(listbox_a.size())]
    # 3D STL 경로 리스트
    dir_b = [listbox_b.get(i) for i in range(listbox_b.size())]

    ### 2D PNG 파일 전처리 ###
    # : dir_b 리스트 입력받아 변수에 이미지 리스트 저장
    print("   1. 2D PNG File Pre-processing . . .", end="")
    base_path_list = process_images(dir_a)[0]
    # ㄴ 이미지 리스트로 받았었는데, 그냥 하나만 넘기는걸로 단순화해서 [0]
    # ㄴ 0.91 - 경로 리스트 반환하는거로 바꿈
    print("  Done")

    ### 3D STL 파일 전처리 ###
    #  : dir_b 리스트 입력받아 temp 폴더 내부에 전처리 결과 저장
    print("   2. 3D STL File Pre-processing . . .", end="")
    crown_path_list = process_stl_files(dir_b)
    print("  Done")

    ### 크라운 배치 자동화 ###
    # var_desc 로 내림차순, 오름차순으로 할지 값 호출
    # var_size 로 Scale factor 값 호출
    # var_rotate 로 회전각도 파라미터 값 호출
    # 두 이미지 간 비율조정 값 계산
    # 우선 코드 정상실행을 확인한 뒤에 변경할 것

    ### 이미지 크기 순 정렬 ###
    # 각 이미지의 크기와 경로를 튜플로 묶어 리스트 생성
    image_size_and_paths = [(get_image_size(path), path) for path in crown_path_list]

    # 이미지 크기(너비 x 높이)에 따라 정렬 (reverse=True 는 내림차순 정렬)
    image_size_and_paths.sort(key=lambda x: x[0][0] * x[0][1], reverse=bool(int(var_desc)))#True)

    # 정렬된 리스트에서 이미지 경로만 추출
    crown_path_list = [path for _, path in image_size_and_paths]

    ### 크라운 배치 순서 표시 ###
    print("   3. Crown Set Order")
    for i,c in enumerate(crown_path_list):
        print("      ",i+1,"번 :",c)

    ### 이미지 호출 및 전처리 ###
    print("   4. Crown Batch Processing . . . It takes time")

    # 이미지를 불러옵니다.
    base_image = Image.open(base_path_list).convert("L")  # 베이스 이미지를 그레이스케일로 변환
    implant_images = [Image.open(path).convert("L") for path in crown_path_list]  # 임플란트 이미지들을 그레이스케일로 변환

    # 임플란트 이미지의 크기를 조정합니다.
    #scale_factor = 1/5  # 크기를 조정할 비율을 설정합니다. # 고정 스케일 팩터 코드
    scale_factor = 1/float(var_size) # 파라미터로 받는 Scale Factor
    resized_implant_images = resize_images(implant_images, scale_factor)

    # 동적 이미지 처리
    plt.ion()

    # 임플란트를 베이스 이미지에 배치합니다.
    #placed_base_image, placed_implants_info = place_implants(base_image, resized_implant_images)
    _, place_img, number_img = place_implants_and_numbers(base_image, resized_implant_images, int(var_rotate))

    ### 이미지 출력 ###

    # 첫 번째 이미지 (임플란트 배치)
    plt.subplot(1, 2, 1)  # 1행 2열의 첫 번째 서브플롯
    plt.imshow(place_img, cmap='gray')
    plt.title("Crown Arrangement")
    plt.axis('off')

    # 두 번째 이미지 (임플란트 번호)
    plt.subplot(1, 2, 2)  # 1행 2열의 두 번째 서브플롯
    plt.imshow(number_img, cmap='gray')
    plt.title("Crown Number")
    plt.axis('off')

    # 전체 이미지 표시
    plt.show()


# Tkinter 윈도우 생성
root = tk.Tk()

# 윈도우 제목 설정
root.title("Crown-Base Batcher v0.94")
# 중의적 의미 ㅋㅋ 배치 - Batch

# 윈도우 사이즈 및 위치 설정
root.geometry("1280x600+0+0")

# 윈도우 닫기 버튼 이벤트 핸들링
root.protocol("WM_DELETE_WINDOW", on_closing)

# 폰트 설정
custom_font = font.Font(family="Helvetica", size=12, weight="bold")

# 아이콘 설정
# PyInstaller가 생성한 임시 디렉토리에서 .ico 파일 찾기
if getattr(sys, 'frozen', False):
    # 실행 파일 모드인 경우
    application_path = sys._MEIPASS
else:
    # 일반 Python 인터프리터 모드인 경우
    application_path = os.path.dirname(os.path.abspath(__file__))

ico_path = os.path.join(application_path, 'Crown-Base-Batcher.ico')
root.iconbitmap(ico_path)

# 파일 선택 프레임 (A Files와 B Files를 포함하는 프레임)
file_frame = tk.Frame(root)
file_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# "A Files" 프레임
frame_a = tk.Frame(file_frame)
frame_a.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

label_a = tk.Label(frame_a, text="Base Data (PNG)", font=custom_font)
label_a.pack()

button_frame_a = tk.Frame(frame_a)
button_frame_a.pack(fill=tk.X)
open_button_a = tk.Button(button_frame_a, text="  Select  ", command=open_a_files)
open_button_a.pack(side=tk.LEFT)
reset_button_a = tk.Button(button_frame_a, text="  Reset  ", command=reset_a_files)
reset_button_a.pack(side=tk.LEFT)

listbox_a = tk.Listbox(frame_a, width=30, height=5)
listbox_a.pack(fill=tk.BOTH, expand=True)

# "B Files" 프레임
frame_b = tk.Frame(file_frame)
frame_b.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

label_b = tk.Label(frame_b, text="Crown Data (STL)", font=custom_font)
label_b.pack()

button_frame_b = tk.Frame(frame_b)
button_frame_b.pack(fill=tk.X)
open_button_b = tk.Button(button_frame_b, text="  Select  ", command=open_b_files)
open_button_b.pack(side=tk.LEFT)
reset_button_b = tk.Button(button_frame_b, text="  Reset  ", command=reset_b_files)
reset_button_b.pack(side=tk.LEFT)

listbox_b = tk.Listbox(frame_b, width=30, height=5)
listbox_b.pack(fill=tk.BOTH, expand=True)

# 옵션 인터페이스 프레임
options_frame = tk.Frame(root)
options_frame.pack(fill=tk.X, padx=10, pady=5)

# 라디오 버튼 라벨
radio_label = tk.Label(options_frame, text="배치순서 :")
radio_label.pack(side=tk.LEFT, padx=5)

# 라디오 버튼
var_desc = tk.StringVar(value=True)  # 'var_desc'는 선택된 라디오 버튼의 값을 저장합니다. 기본값을 "option1"으로 설정합니다.
radio_button1 = tk.Radiobutton(options_frame, text="큰것부터", variable=var_desc, value=True)
# 'radio_button1'을 생성합니다. 'text'는 버튼 옆에 표시될 텍스트입니다.
# 'variable'은 이 라디오 버튼 그룹의 선택값을 저장할 변수를 지정합니다.
# 'value'는 이 버튼이 선택됐을 때 'var_desc'에 저장될 값입니다.
radio_button1.pack(side=tk.LEFT, padx=2)  # 버튼을 왼쪽 정렬로 배치하고, 좌우에 패딩을 5픽셀 추가합니다.

radio_button2 = tk.Radiobutton(options_frame, text="작은것부터", variable=var_desc, value=False)
# 'radio_button2'를 생성합니다. 설정은 'radio_button1'과 유사하지만, 선택될 때의 값이 "option2"입니다.
radio_button2.pack(side=tk.LEFT, padx=2)  # 이 버튼도 왼쪽 정렬로 배치하고, 좌우에 패딩을 5픽셀 추가합니다.

# 입력 필드 1 라벨
entry_label1 = tk.Label(options_frame, text="             크기배수 :")
entry_label1.pack(side=tk.LEFT, padx=5)

# 입력 필드 1
vcmd1 = root.register(validate_number) # 입력값 검증 (숫자만)
var_size = tk.StringVar(value=5) # 기본값 설정
entry1 = tk.Entry(options_frame, textvariable=var_size, validate="key", validatecommand=(vcmd1, '%P'), width=15)
entry1.pack(side=tk.LEFT, padx=5)

# 입력 필드 2 라벨
entry_label2 = tk.Label(options_frame, text="             회전각도 :")
entry_label2.pack(side=tk.LEFT, padx=5)

# 입력 필드 2
vcmd2 = root.register(validate_number)
val_rotate = tk.StringVar(value=60)
entry2 = tk.Entry(options_frame, textvariable=val_rotate, validate="key", validatecommand=(vcmd2, '%P'), width=15)
entry2.pack(side=tk.LEFT, padx=5)


# 실행 버튼
execute_button = tk.Button(root, text="Execute", command=lambda: execute_action(var_desc.get(), var_size.get(), val_rotate.get()))
execute_button.pack(fill=tk.X, padx=10, pady=5)


# 로그를 표시할 Text 위젯 생성
text_widget = tk.Text(root, height=10, width=50)
text_widget.pack(fill=tk.BOTH, expand=True)

# Print 구문의 출력을 Text 위젯으로 재지정
sys.stdout = TextRedirector(text_widget)

# 윈도우 실행
root.mainloop()
