import numpy as np

PAYLOAD_LEN = 512

print(f"--- Payload Generator ---")
print(f"Generating {PAYLOAD_LEN} random complex symbols...")

payload = np.random.randn(PAYLOAD_LEN) + 1j * np.random.randn(PAYLOAD_LEN)

payload /= np.sqrt(np.mean(np.abs(payload)**2))

np.savez('gt_data.npz', gt_payload=payload)

print(f"Success! Ground truth saved to 'gt_data.npz'.")