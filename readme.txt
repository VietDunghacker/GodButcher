Giải thích rõ hơn về code trong file main.py:
- bad_words là một cái dictionary, key là tên file và value là một set chứa những bad word trong file đó.
Các bad word sẽ có thể nhiều hơn 1 từ, cần phải xử lý kỹ lưỡng.
- function negative_features:
Đây là function chính, các hàm mà anh em viết là helper function và sẽ được sử dụng trong function này. Lưu ý là các helper function sẽ return dưới dạng dictionary.
Input của function này, sent, là một đoạn string chưa qua xử lý gì hết.
words là một list của word sau khi đã tokenize đoạn string đó ra.
tag_words là list sau khi đã thêm tag vào các từ trong words.

