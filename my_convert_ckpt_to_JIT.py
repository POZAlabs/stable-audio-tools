import argparse
from pathlib import Path

import torch


def restore_weight_from_weight_norm(state_dict):
    """
    Weight Normalization이 적용된 state_dict에서 원래의 weight를 복원합니다.
    '_g'와 '_v' 접미사를 가진 파라미터를 찾아 원래의 'weight'로 재계산합니다.
    """
    restored_state_dict = {}
    processed_keys = set()

    # 1. Weight Normalization 복원 처리
    for key in list(state_dict.keys()):
        if key.endswith('_g'):
            # 예: 'layers.0.conv.weight_g' -> 'layers.0.conv.weight'
            base_name = key[:-2]
            v_key = base_name + '_v'
            if v_key in state_dict:
                weight_g = state_dict[key]
                weight_v = state_dict[v_key]

                dims = tuple(range(1, weight_v.dim()))
                norm_v = torch.norm(weight_v, p=2, dim=dims, keepdim=True)
                restored_weight = weight_g * (weight_v / (norm_v + 1e-12))

                restored_state_dict[base_name] = restored_weight
                processed_keys.add(key)
                processed_keys.add(v_key)

    # 2. 나머지 파라미터 복사
    for key, value in state_dict.items():
        if key not in processed_keys:
            restored_state_dict[key] = value

    return restored_state_dict


def update_jit_model_weights(original_jit_path: str, trained_ckpt_path: str, output_path: str):
    """
    JIT 모델의 파라미터를 stable-audio-tools 체크포인트의 EMA 가중치로 교체하여 저장합니다.
    """
    print(f"1. 원본 JIT 모델 로드 중: '{original_jit_path}'")
    jit_model = torch.jit.load(original_jit_path, map_location='cpu')
    jit_model.eval()

    print(f"2. 학습된 체크포인트 로드 중: '{trained_ckpt_path}'")
    checkpoint = torch.load(trained_ckpt_path, map_location='cpu')

    if 'state_dict' not in checkpoint:
        raise KeyError("Checkpoint does not contain a 'state_dict' key.")

    full_state_dict = checkpoint['state_dict']

    print("3. EMA 모델 가중치 추출 및 키 정리 중...")
    # 'autoencoder_ema.ema_model.' 접두사 제거
    ema_prefix = "autoencoder_ema.ema_model."

    ema_state_dict = {}
    for key, value in full_state_dict.items():
        if key.startswith(ema_prefix):
            new_key = key[len(ema_prefix):]
            ema_state_dict[new_key] = value

    # EMA가 없는 경우 일반 모델 가중치 사용
    if not ema_state_dict:
        print("   - 경고: 'autoencoder_ema.ema_model.' 접두사를 가진 EMA 가중치를 찾을 수 없습니다.")
        print("   - 'autoencoder.' 접두사를 가진 일반 모델 가중치로 대신합니다.")
        prefix_to_remove = "autoencoder."
        ema_state_dict = {
            key[len(prefix_to_remove):]: value
            for key, value in full_state_dict.items()
            if key.startswith(prefix_to_remove)
        }
        if not ema_state_dict:
            raise KeyError("체크포인트에서 'autoencoder_ema.ema_model.' 또는 'autoencoder.'로 시작하는 가중치를 찾을 수 없습니다.")
    else:
        print(f"   - '{ema_prefix}' 접두사를 가진 EMA 가중치를 성공적으로 추출했습니다.")

    print("4. Weight Normalization 파라미터 복원 시작...")
    restored_state_dict = restore_weight_from_weight_norm(ema_state_dict)

    print("5. JIT 모델에 복원된 파라미터 주입 중...")
    try:
        jit_model.load_state_dict(restored_state_dict)
        print("   - 파라미터 주입 성공!")
    except RuntimeError as e:
        print(f"   - 에러: 파라미터 주입 실패. JIT 모델과 체크포인트의 구조가 일치하는지 확인하세요.")
        print(f"   - 상세 에러: {e}")
        print("\n--- JIT Model Keys (Top 20) ---")
        print(list(jit_model.state_dict().keys())[:20])
        print("\n--- Checkpoint Keys (Top 20) after processing ---")
        print(list(restored_state_dict.keys())[:20])
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"6. 업데이트된 JIT 모델 저장 중: '{output_path}'")
    jit_model.save(output_path)

    print("\n작업 완료!")
    print(f"최종 모델이 '{output_path}'에 성공적으로 저장되었습니다.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="JIT 모델의 가중치를 Weight Normalization이 적용된 체크포인트로 업데이트합니다.")
    parser.add_argument("-oj", "--original_jit", type=str, required=True, help="구조를 가져올 원본 JIT 모델 파일 경로 (*.pt)")
    parser.add_argument("-tc", "--trained_ckpt", type=str, required=True, help="업데이트할 가중치가 포함된 체크포인트 파일 경로 (*.ckpt)")
    parser.add_argument("-o", "--output", type=str, required=True, help="가중치가 업데이트된 새로운 JIT 모델을 저장할 파일 경로 (*.pt)")
    args = parser.parse_args()
    update_jit_model_weights(args.original_jit, args.trained_ckpt, args.output)
