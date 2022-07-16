# tracking-car-using-aruco-
chương trình gồm 1 folder chứa ảnh và 7 file code, mỗi file có 1 chức năng khác nhau 

folder save_images chứa ảnh cho mục đích calib camera. trong folder save_images hiện tại chứa ảnh chụp checkboard(9x6) bằng camera c280. các bạn sử dụng camera khác thì trước khi sử dụng hãy xóa hết ảnh trong thư mục này đi. 

file 1_save_frames.py với mục đích lưu ảnh checkboard ( các ảnh checkboard sẽ tự động được lưu ở folder save_image ). khi chương trình được chạy, các bạn ấn dấu “cách” để chụp và lưu ảnh. Lưu ý nên chụp ít nhất 30 ảnh 

file 2_artag_opencv.py : sau khi chạy xong file 1 các bạn kiểm tra xem đã có ảnh checkboard trong folder save_image chưa. Nếu đã có ảnh thì bắt đầu chạy 2_artag_opencv.py với mục đích lấy các thông số của camera và các hệ số biến dạng cảu ảnh 

file 4_resold_motion_blur.py có mục địch khử hiện tượng nhòe ảnh. khi chạy file này sẽ có 1 bảng chứa các thông số xuất hiện. Các bạn điều chỉnh các thông số trong bảng đến khi hết hiện tượng motion blur nhé 

file 3_tọa độ map.py  chạy file 4 trước file 3 ( vì lý do kỹ thuật ). File 3 với mục đích lấy tọa độ map . các bạn cần đảm bảo 4 aruco ở 4 góc sa bàn đều nằm trong phạm vi camera

file 5_main.py lấy tọa độ và vận tốc của xe và trả về 1 file main.txt chưa vận tốc và vị trí 

file 6_using_other_cmd_draw_velocity_realtime.py và file 7_draw_position_real_time.py : file 6 và 7 có thể chạy đồng thời với file 5 để vẽ đồ thị vận tốc và đồ thị vị trí. 

Lưu ý 
các bạn cần in checkboard ra 1 tờ giấy A4, sau đó đo kích thước của 1 ô vuông trên checkboard với thứ nguyên là mm sau đó nhập thông số đó vào dòng số 9 file 2 

source code được đặt trên github bất cứ ai cũng có thể tải về và sử dụng 

các thư viện cần thiết được đặt trong file prerequisite.txt 

aruco có id 25, 35, 45, 72 các bạn để theo thứ tự tăng dần vào 4 góc sa bàn nhé. aruco 35 là gốc tọa độ nhé, (25, 35) là trục x, (35,45) là trục y
aruco dán trên xe: các bạn vào 5_main.py sửa dòng 155 thành id mà các bạn mong muốn nhé (hiện tại trong code mình đang để là 25)

khi chạy file 6, 7 để lấy vận tốc và vị trí (chạy termial trên máy của các bạn không phải trên môi trường ảo) các bạn cài giúp mình matplotlib, pandas, DateTime, dateutil trực tiếp trên máy nhé không phải trong môi trường ảo (global)

