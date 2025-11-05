# This file is a multiplex/multi-channel version of the registration pipeline.
# It saves all channels from the warped Orion image for each patch.
# The output Orion patch will have shape (patch_size, patch_size, num_channels).

# ... existing code ...
# (The rest of the file is identical to registration_pipeline.py, except for the following changes:)

# In the DatasetPreparator class, update create_training_pairs and _extract_patches:

class DatasetPreparatorMultiplex(DatasetPreparator):
    """Prepares training datasets from registered images (multi-channel version)."""
    def create_training_pairs(self, registration_results: List[Dict]) -> str:
        pairs_dir = pathlib.Path(self.config.output_dir) / "training_pairs_multiplex"
        pairs_dir.mkdir(exist_ok=True)
        successful_results = [r for r in registration_results if r['success']]
        logger.info(f"Creating multiplex training pairs from {len(successful_results)} successful registrations")
        pair_count = 0
        for result in successful_results:
            he_img = result['he_img']
            warped_orion = result['warped_orion']
            core_id = result['core_id']
            # Create patches
            patches = self._extract_patches(he_img, warped_orion, core_id)
            for i, (he_patch, orion_patch) in enumerate(patches):
                if self._is_valid_patch(orion_patch):
                    patch_id = f"{core_id}_patch_{i:04d}"
                    if self.config.save_npy_pairs:
                        np.save(pairs_dir / f"{patch_id}_HE.npy", he_patch)
                        np.save(pairs_dir / f"{patch_id}_ORION.npy", orion_patch)
                    pair_count += 1
        logger.info(f"Created {pair_count} multiplex training pairs")
        return str(pairs_dir)
    def _extract_patches(self, he_img: np.ndarray, orion_img: np.ndarray, core_id: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        patches = []
        for y in range(0, he_img.shape[0] - self.config.patch_size + 1, self.config.stride):
            for x in range(0, he_img.shape[1] - self.config.patch_size + 1, self.config.stride):
                he_patch = he_img[y:y + self.config.patch_size, x:x + self.config.patch_size]
                # If orion_img is multi-channel (C, H, W) or (H, W, C), handle both
                if orion_img.ndim == 3 and (orion_img.shape[0] == he_img.shape[0] and orion_img.shape[1] == he_img.shape[1]):
                    # (H, W, C) format
                    orion_patch = orion_img[y:y + self.config.patch_size, x:x + self.config.patch_size, :]
                elif orion_img.ndim == 3 and (orion_img.shape[1] == he_img.shape[0] and orion_img.shape[2] == he_img.shape[1]):
                    # (C, H, W) format
                    orion_patch = orion_img[:, y:y + self.config.patch_size, x:x + self.config.patch_size]
                    orion_patch = np.transpose(orion_patch, (1, 2, 0))
                else:
                    # Fallback: treat as single channel
                    orion_patch = orion_img[y:y + self.config.patch_size, x:x + self.config.patch_size]
                patches.append((he_patch, orion_patch))
        return patches

# In the RegistrationPipeline class, use DatasetPreparatorMultiplex
class RegistrationPipelineMultiplex(RegistrationPipeline):
    def __init__(self, config: RegistrationConfig):
        self.config = config
        self.registrar = VALISRegistrar(config)
        self.preparator = DatasetPreparatorMultiplex(config)
        self.qc = QualityController(config)
        self.output_path = pathlib.Path(config.output_dir)
        self.output_path.mkdir(exist_ok=True)
        (self.output_path / "registered_images").mkdir(exist_ok=True)
        (self.output_path / "quality_plots").mkdir(exist_ok=True)
        (self.output_path / "training_pairs_multiplex").mkdir(exist_ok=True)

# In main(), instantiate RegistrationPipelineMultiplex
if __name__ == "__main__":
    config = RegistrationConfig(
        input_dir="/path/to/your/image/pairs",
        output_dir="./registration_output_multiplex",
        he_suffix="_HE.tif",
        orion_suffix="_Orion.tif",
        patch_size=256,
        stride=256,
        num_workers=4
    )
    pipeline = RegistrationPipelineMultiplex(config)
    results = pipeline.run()
    print("Multiplex Registration Pipeline Results:")
    print(f"Total pairs: {results['total_image_pairs']}")
    print(f"Successful registrations: {results['successful_registrations']}")
    print(f"Success rate: {results['success_rate']:.2%}")
    print(f"Multiplex training pairs directory: {results['training_pairs_directory']}") 