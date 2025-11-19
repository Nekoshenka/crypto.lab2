import os
import math
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy import stats


class LSBSteganography:
    def __init__(self):
        self.images_folder = 'imgs'
        self.results_folder = 'results'
        os.makedirs(self.results_folder, exist_ok=True)

    def load_image(self, image_path):
        img = Image.open(image_path)
        return np.array(img), img

    def save_image(self, image_array, filename):
        img = Image.fromarray(image_array.astype('uint8'))
        img.save(os.path.join(self.results_folder, filename))

    def text_to_bits(self, text):
        text_bytes = text.encode('utf-8')

        length = len(text_bytes)
        length_bits = []
        for i in range(4):
            length_bits.extend([int(b) for b in format((length >> (8 * (3 - i))) & 0xFF, '08b')])

        data_bits = []
        for byte in text_bytes:
            data_bits.extend([int(b) for b in format(byte, '08b')])

        return length_bits + data_bits

    def bits_to_text(self, bits):
        if len(bits) < 32:
            return ""

        length = 0
        for i in range(4):
            byte_bits = bits[i * 8:(i + 1) * 8]
            if len(byte_bits) != 8:
                return ""
            byte_val = int(''.join(str(b) for b in byte_bits), 2)
            length = (length << 8) | byte_val

        total_bits_needed = 32 + length * 8
        if len(bits) < total_bits_needed:
            return ""

        text_bytes = bytearray()
        for i in range(32, total_bits_needed, 8):
            byte_bits = bits[i:i + 8]
            if len(byte_bits) != 8:
                return ""
            byte_val = int(''.join(str(b) for b in byte_bits), 2)
            text_bytes.append(byte_val)

        try:
            return text_bytes.decode('utf-8')
        except:
            return ""

    def calculate_psnr(self, original, stego):
        mse = np.mean((original - stego) ** 2)

        if mse == 0:
            return float('inf')  # случай полного совпадения

        return 20 * math.log10(255.0 / math.sqrt(mse))

    def encode_lsb(self, image_array, text, payload_ratio=1.0):
        height, width, channels = image_array.shape

        all_bits = self.text_to_bits(text)
        total_bits = len(all_bits)

        max_capacity = height * width * 3
        available_bits = int(max_capacity * payload_ratio)

        if total_bits > available_bits:
            available_data_bits = available_bits - 32  # минус 32 бита на длину
            max_text_length = available_data_bits // 8
            if max_text_length <= 0:
                raise ValueError("Изображение слишком маленькое для встраивания сообщения")

            truncated_text = text[:max_text_length]
            all_bits = self.text_to_bits(truncated_text)
            total_bits = len(all_bits)
            print(f"Текст обрезан до {max_text_length} символов")

        print(f"Встраивание {total_bits} битов (ёмкость: {max_capacity})")

        stego_image = image_array.copy()
        bit_index = 0

        for i in range(height):
            for j in range(width):
                for k in range(3):
                    if bit_index < total_bits:
                        pixel_value = stego_image[i, j, k]
                        new_pixel = (pixel_value & 0xFE) | all_bits[bit_index]
                        # new_pixel = (pixel_value & 0x7F) | (all_bits[bit_index] << 7)  # MSB lol
                        stego_image[i, j, k] = new_pixel
                        bit_index += 1
                    else:
                        break
                if bit_index >= total_bits:
                    break
            if bit_index >= total_bits:
                break

        return stego_image, total_bits

    def decode_lsb(self, stego_array, total_bits=None):
        height, width, channels = stego_array.shape

        if total_bits is None:
            total_bits = height * width * 3

        bits = []
        bit_count = 0

        for i in range(height):
            for j in range(width):
                for k in range(3):
                    if bit_count >= total_bits:
                        break
                    pixel_value = stego_array[i, j, k]
                    lsb = pixel_value & 1
                    bits.append(lsb)
                    bit_count += 1
                if bit_count >= total_bits:
                    break
            if bit_count >= total_bits:
                break

        return self.bits_to_text(bits)

    def create_heatmap(self, original, stego, image_name):
        difference = np.abs(original.astype(float) - stego.astype(float))

        diff_avg = np.mean(difference, axis=2)

        plt.figure(figsize=(12, 10))

        im = plt.imshow(diff_avg, cmap='hot', vmin=0, vmax=1)
        plt.title(f'Карта разности для {image_name}')
        plt.axis('off')
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('Величина изменения', rotation=270, labelpad=15)

        heatmap_filename = f"heatmap_{image_name}.png"
        plt.savefig(os.path.join(self.results_folder, heatmap_filename), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"Карта разности сохранена: {heatmap_filename}")

        return diff_avg

    def analyze_differences(self, original, stego, image_name):
        np.abs(original.astype(float) - stego.astype(float))

        psnr = self.calculate_psnr(original, stego)

        ssim_r = ssim(original[:, :, 0], stego[:, :, 0], data_range=255)
        ssim_g = ssim(original[:, :, 1], stego[:, :, 1], data_range=255)
        ssim_b = ssim(original[:, :, 2], stego[:, :, 2], data_range=255)
        ssim_avg = (ssim_r + ssim_g + ssim_b) / 3

        diff_avg = self.create_heatmap(original, stego, image_name)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Оригинальное изображение')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(stego)
        axes[0, 1].set_title('Стего-изображение')
        axes[0, 1].axis('off')

        im = axes[0, 2].imshow(diff_avg, cmap='hot', vmin=0, vmax=1)
        axes[0, 2].set_title('Карта разности')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2])

        axes[1, 0].hist(original[:, :, 0].flatten(), bins=50, alpha=0.7, label='Оригинал', color='red')
        axes[1, 0].hist(stego[:, :, 0].flatten(), bins=50, alpha=0.7, label='Стего-', color='darkred')
        axes[1, 0].set_title('Гистограмма красного')
        axes[1, 0].legend()

        axes[1, 1].hist(original[:, :, 1].flatten(), bins=50, alpha=0.7, label='Оригинал', color='green')
        axes[1, 1].hist(stego[:, :, 1].flatten(), bins=50, alpha=0.7, label='Стего-', color='darkgreen')
        axes[1, 1].set_title('Гистограмма зелёного')
        axes[1, 1].legend()

        axes[1, 2].hist(original[:, :, 2].flatten(), bins=50, alpha=0.7, label='Оригинал', color='blue')
        axes[1, 2].hist(stego[:, :, 2].flatten(), bins=50, alpha=0.7, label='Стего-', color='darkblue')
        axes[1, 2].set_title('Гистограмма синего')
        axes[1, 2].legend()

        plt.suptitle(f'Анализ для {image_name}\nPSNR: {psnr:.2f} dB, SSIM: {ssim_avg:.4f}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_folder, f'analysis_{image_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()

        return psnr, ssim_avg

    def chi_square_test(self, image_array):
        results = []

        for channel in range(3):
            channel_data = image_array[:, :, channel].flatten()

            hist, _ = np.histogram(channel_data, bins=256, range=(0, 255))

            chi_square = 0
            degrees_of_freedom = 0

            for k in range(128):
                n2k = hist[2 * k]
                n2k1 = hist[2 * k + 1]

                total = n2k + n2k1
                if total > 0:
                    expected = total / 2
                    chi_square += ((n2k - expected) ** 2) / expected
                    chi_square += ((n2k1 - expected) ** 2) / expected
                    degrees_of_freedom += 1

            if degrees_of_freedom > 0:
                p_value = 1 - stats.chi2.cdf(chi_square, degrees_of_freedom)
            else:
                p_value = 1.0

            results.append({
                'channel': ['R', 'G', 'B'][channel],
                'chi_square': chi_square,
                'p_value': p_value,
                'detected': p_value < 0.05
            })

        return results

    def run_experiment(self, image_name, text, payload_ratios=[0.25, 0.5, 0.75, 1.0]):
        print(f"\n=== {image_name} ===")

        image_path = os.path.join(self.images_folder, image_name)
        try:
            original_array, original_img = self.load_image(image_path)
        except Exception as e:
            print(f"Ошибка загрузки изображения: {e}")
            return []

        height, width, channels = original_array.shape
        print(f"Размер изображения: {width}х{height}, Каналы: {channels}")
        print(f"Максимальная ёмкость: {height * width * channels} битов")

        results = []

        for ratio in payload_ratios:
            print(f"\nДля Payload ratio: {ratio}")

            try:
                stego_array, bits_used = self.encode_lsb(original_array, text, ratio)

                stego_filename = f"stego_{ratio}_{image_name}"
                self.save_image(stego_array, stego_filename)

                extracted_text = self.decode_lsb(stego_array, bits_used)

                psnr, ssim_val = self.analyze_differences(original_array, stego_array, f"{ratio}_{image_name}")

                chi_results = self.chi_square_test(stego_array)

                result = {
                    'payload_ratio': ratio,
                    'bits_used': bits_used,
                    'max_capacity': height * width * 3,
                    'psnr': psnr,
                    'ssim': ssim_val,
                    'original_text': text,
                    'extracted_text': extracted_text,
                    'text_correct': extracted_text == text,
                    'chi_square_results': chi_results
                }

                results.append(result)

                print(f"Использовано битов: {bits_used}")
                print(f"PSNR: {psnr:.2f} dB")
                print(f"SSIM: {ssim_val:.4f}")
                print(f"Корректность текста: {extracted_text == text}")

                if not extracted_text == text:
                    print(f"Ожидается: {text}")
                    # print(f"Ожидается: то")
                    print(f"Получено: {extracted_text}")
                    # print(f"Получено: не то")

                for chi in chi_results:
                    status = "Обнаружено" if chi['detected'] else "Не обнаружено"
                    # print(f"Канал {chi['channel']}: p-value = {chi['p_value']:.8f} ({status})")
                    print(f"Канал {chi['channel']}: p-value = {chi['p_value']} ({status})")

            except Exception as e:
                print(f"Ошибка: {e}")
                continue

        return results

    def generate_report(self, all_results):
        print("\n===== Метрики =====\n")

        for image_name, results in all_results.items():
            print(f"Результаты {image_name}:")

            for result in results:
                print(f"   Payload ratio: {result['payload_ratio']}")
                print(f"   Биты: {result['bits_used']}/{result['max_capacity']} "
                      f"({result['bits_used'] / result['max_capacity'] * 100:.4f}%)")
                print(f"   PSNR: {result['psnr']:.2f} dB")
                print(f"   SSIM: {result['ssim']:.4f}")
                print(f"   Корректность текста: {result['text_correct']}")

                detection_count = sum(1 for chi in result['chi_square_results'] if chi['detected'])
                print(f"   Обнаружение (хи-квадрат): {detection_count}/3\n")

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        for image_name, image_results in all_results.items():
            ratios_img = [r['payload_ratio'] for r in image_results]
            psnrs_img = [r['psnr'] for r in image_results]
            plt.plot(ratios_img, psnrs_img, 'o-', label=image_name, markersize=6, linewidth=2)

        plt.xlabel('Payload Ratio')
        plt.ylabel('PSNR (dB)')
        plt.title('Зависимость PSNR от степени загрузки')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        for image_name, image_results in all_results.items():
            ratios_img = [r['payload_ratio'] for r in image_results]
            ssims_img = [r['ssim'] for r in image_results]
            plt.plot(ratios_img, ssims_img, 'o-', label=image_name, markersize=6, linewidth=2)

        plt.ylim(0.9, 1.01)
        plt.xlabel('Payload Ratio')
        plt.ylabel('SSIM')
        plt.title('Зависимость SSIM от степени загрузки')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        for image_name, image_results in all_results.items():
            ratios_img = [r['payload_ratio'] for r in image_results]
            detection_rates = [sum(1 for chi in r['chi_square_results'] if chi['detected']) / 3
                               for r in image_results]
            plt.plot(ratios_img, detection_rates, 'o-', label=image_name, markersize=6, linewidth=2)

        plt.xlabel('Payload Ratio')
        plt.ylabel('Вероятность обнаружения')
        plt.title('Зависимость обнаружимости от степени загрузки')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        for image_name, image_results in all_results.items():
            ratios_img = [r['payload_ratio'] for r in image_results]
            usage = [r['bits_used'] / r['max_capacity'] for r in image_results]
            plt.plot(ratios_img, usage, 'o-', label=image_name, markersize=6, linewidth=2)

        plt.xlabel('Payload Ratio')
        plt.ylabel('Использование ёмкости')
        plt.title('Использование доступной ёмкости')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_folder, 'summary_report.png'), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Графики сохранены в: summary_report.png")


stego = LSBSteganography()

if not os.path.exists(stego.images_folder):
    print(f"Папка '{stego.images_folder}' не найдена")
    sys.exit(0)

# test_text = "1"  # минимальное
# test_text = "LSB1"*660*660*3  # достаточное большое
# test_text = "1"*660*660*3  # смешное
test_text = "9p754[0u34-i]0'lce;98ce5[9lzw4k48tub9py854c09serulte;0d4t8;e04;ct0e4;tv;0edchulgtttse;0" * 10000  # много.

test_images = [
    'gradient.png',
    'checkerboard.png',
    'noise_texture.png',
    'image.png'
]

available_images = []
for img in test_images:
    img_path = os.path.join(stego.images_folder, img)
    if os.path.exists(img_path):
        available_images.append(img)
    else:
        print(f"Файл {img} не найден")
if not available_images:
    sys.exit()

print(f"Передаваемое сообщение: {test_text}")

all_results = {}

for image_name in available_images:
    try:
        results = stego.run_experiment(image_name, test_text)
        all_results[image_name] = results

    except Exception as e:
        print(f"Ошибка с {image_name}: {e}")
        continue

if all_results:
    stego.generate_report(all_results)
