"""Testes unitários para stage2_leaf_seg.py (F2)."""

import numpy as np
import pytest
import cv2

from plant_disease.stage2_leaf_seg import (
    LeafSegmentationConfig,
    segment_leaf,
    segment_leaf_otsu,
    extract_largest_contour,
    apply_mask,
    LeafSegmenter,
)


class TestSegmentLeafValidation:
    """Testes de validação de inputs para segment_leaf."""

    def test_segment_leaf_none_input(self):
        """Deve retornar None para input None."""
        result = segment_leaf(None)
        assert result is None

    def test_segment_leaf_invalid_type(self):
        """Deve retornar None para input não-ndarray."""
        result = segment_leaf("not_an_image")
        assert result is None

    def test_segment_leaf_wrong_channels(self):
        """Deve retornar None para imagem com número errado de canais."""
        wrong_hsv = np.zeros((256, 256, 4), dtype=np.uint8)
        result = segment_leaf(wrong_hsv)
        assert result is None

    def test_segment_leaf_wrong_dtype(self):
        """Deve retornar None para dtype diferente de uint8."""
        wrong_hsv = np.zeros((256, 256, 3), dtype=np.float32)
        result = segment_leaf(wrong_hsv)
        assert result is None


class TestSegmentLeafFunctionality:
    """Testes de funcionalidade para segment_leaf."""

    def test_segment_leaf_valid_hsv(self):
        """Deve retornar máscara e área para HSV válida."""
        hsv = np.zeros((256, 256, 3), dtype=np.uint8)
        hsv[50:150, 50:150] = [50, 100, 100]  # Verde (H=50)

        result = segment_leaf(hsv)
        assert result is not None
        mask, area = result
        assert isinstance(mask, np.ndarray)
        assert isinstance(area, int)
        assert mask.shape == (256, 256)
        assert mask.dtype == np.uint8
        assert area > 0

    def test_segment_leaf_empty_hsv(self):
        """Deve retornar máscara vazia para HSV fora de range."""
        hsv = np.zeros((256, 256, 3), dtype=np.uint8)
        hsv[:, :] = [0, 0, 0]  # Fora do range verde

        result = segment_leaf(hsv)
        assert result is not None
        mask, area = result
        assert area == 0

    def test_segment_leaf_custom_params(self):
        """Deve aceitar parâmetros customizados."""
        hsv = np.zeros((256, 256, 3), dtype=np.uint8)
        hsv[50:150, 50:150] = [50, 100, 100]

        result = segment_leaf(hsv, h_min=40, h_max=60, s_min=50)
        assert result is not None
        mask, area = result
        assert area > 0


class TestSegmentLeafOtsuValidation:
    """Testes de validação de inputs para segment_leaf_otsu."""

    def test_segment_leaf_otsu_none_input(self):
        """Deve retornar None para input None."""
        result = segment_leaf_otsu(None)
        assert result is None

    def test_segment_leaf_otsu_invalid_type(self):
        """Deve retornar None para input não-ndarray."""
        result = segment_leaf_otsu("not_an_image")
        assert result is None

    def test_segment_leaf_otsu_wrong_channels(self):
        """Deve retornar None para imagem com múltiplos canais."""
        wrong_gray = np.zeros((256, 256, 3), dtype=np.uint8)
        result = segment_leaf_otsu(wrong_gray)
        assert result is None

    def test_segment_leaf_otsu_wrong_dtype(self):
        """Deve retornar None para dtype diferente de uint8."""
        wrong_gray = np.zeros((256, 256), dtype=np.float32)
        result = segment_leaf_otsu(wrong_gray)
        assert result is None


class TestSegmentLeafOtsuFunctionality:
    """Testes de funcionalidade para segment_leaf_otsu."""

    def test_segment_leaf_otsu_valid_gray(self):
        """Deve retornar máscara e área para cinza válida."""
        gray = np.zeros((256, 256), dtype=np.uint8)
        gray[50:150, 50:150] = 200  # Região clara

        result = segment_leaf_otsu(gray)
        assert result is not None
        mask, area = result
        assert isinstance(mask, np.ndarray)
        assert isinstance(area, int)
        assert mask.shape == (256, 256)
        assert mask.dtype == np.uint8
        assert area > 0

    def test_segment_leaf_otsu_bimodal(self):
        """Deve funcionar bem com distribuição bimodal."""
        gray = np.zeros((256, 256), dtype=np.uint8)
        gray[0:100, :] = 50  # Fundo escuro
        gray[100:256, :] = 200  # Folha clara

        result = segment_leaf_otsu(gray)
        assert result is not None
        mask, area = result
        assert area > 0


class TestExtractLargestContourValidation:
    """Testes de validação para extract_largest_contour (F2.4)."""

    def test_extract_largest_contour_none_input(self):
        """Deve retornar None para input None."""
        result = extract_largest_contour(None)
        assert result is None

    def test_extract_largest_contour_invalid_type(self):
        """Deve retornar None para input não-ndarray."""
        result = extract_largest_contour("not_an_image")
        assert result is None

    def test_extract_largest_contour_wrong_channels(self):
        """Deve retornar None para imagem com múltiplos canais."""
        wrong_mask = np.zeros((256, 256, 3), dtype=np.uint8)
        result = extract_largest_contour(wrong_mask)
        assert result is None

    def test_extract_largest_contour_empty_mask(self):
        """Deve retornar None para máscara vazia."""
        empty_mask = np.zeros((256, 256), dtype=np.uint8)
        result = extract_largest_contour(empty_mask)
        assert result is None


class TestExtractLargestContourFunctionality:
    """Testes de funcionalidade para extract_largest_contour."""

    def test_extract_largest_contour_single_blob(self):
        """Deve extrair contorno de blob único."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(mask, (128, 128), 50, 255, -1)

        result = extract_largest_contour(mask, fill_contour=True)
        assert result is not None
        refined_mask, area = result
        assert isinstance(refined_mask, np.ndarray)
        assert isinstance(area, float)
        assert area > 0

    def test_extract_largest_contour_multiple_blobs(self):
        """Deve extrair apenas o maior contorno."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 30, 255, -1)  # Pequeno
        cv2.circle(mask, (200, 200), 70, 255, -1)  # Grande

        result = extract_largest_contour(mask, fill_contour=True)
        assert result is not None
        refined_mask, area = result
        assert area > 0


class TestApplyMaskValidation:
    """Testes de validação para apply_mask (F2.5)."""

    def test_apply_mask_none_inputs(self):
        """Deve retornar None para inputs None."""
        result = apply_mask(None, None)
        assert result is None

    def test_apply_mask_none_image(self):
        """Deve retornar None para imagem None."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        result = apply_mask(None, mask)
        assert result is None

    def test_apply_mask_none_mask(self):
        """Deve retornar None para máscara None."""
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        result = apply_mask(image, None)
        assert result is None

    def test_apply_mask_wrong_mask_dtype(self):
        """Deve retornar None para máscara com dtype errado."""
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        wrong_mask = np.zeros((256, 256), dtype=np.float32)
        result = apply_mask(image, wrong_mask)
        assert result is None

    def test_apply_mask_shape_mismatch(self):
        """Deve retornar None quando imagem e máscara têm tamanhos diferentes."""
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        mask = np.zeros((200, 200), dtype=np.uint8)
        result = apply_mask(image, mask)
        assert result is None


class TestApplyMaskFunctionality:
    """Testes de funcionalidade para apply_mask."""

    def test_apply_mask_bgr_image(self):
        """Deve aplicar máscara a imagem BGR."""
        image = np.ones((256, 256, 3), dtype=np.uint8) * 128
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[50:150, 50:150] = 255

        result = apply_mask(image, mask)
        assert result is not None
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_apply_mask_gray_image(self):
        """Deve aplicar máscara a imagem em tons de cinza."""
        image = np.ones((256, 256), dtype=np.uint8) * 128
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[50:150, 50:150] = 255

        result = apply_mask(image, mask)
        assert result is not None
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_apply_mask_preserves_masked_region(self):
        """Deve preservar região dentro da máscara."""
        image = np.full((256, 256, 3), 100, dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[50:150, 50:150] = 255

        result = apply_mask(image, mask)
        assert result is not None
        assert np.any(result[50:150, 50:150] > 0)
        assert np.all(result[0:50, 0:50] == 0)


class TestLeafSegmenterClass:
    """Testes para a classe LeafSegmenter."""

    def test_leaf_segmenter_default_config(self):
        """Deve inicializar com configuração padrão."""
        segmenter = LeafSegmenter()
        assert isinstance(segmenter.config, LeafSegmentationConfig)

    def test_leaf_segmenter_custom_config(self):
        """Deve aceitar configuração customizada."""
        config = LeafSegmentationConfig(h_min=30, h_max=90)
        segmenter = LeafSegmenter(config)
        assert segmenter.config.h_min == 30
        assert segmenter.config.h_max == 90

    def test_leaf_segmenter_run_hsv(self):
        """Deve executar run_hsv corretamente."""
        hsv = np.zeros((256, 256, 3), dtype=np.uint8)
        hsv[50:150, 50:150] = [50, 100, 100]

        segmenter = LeafSegmenter()
        result = segmenter.run_hsv(hsv)
        assert result is not None
        mask, area = result
        assert area > 0

    def test_leaf_segmenter_run_otsu(self):
        """Deve executar run_otsu corretamente."""
        gray = np.zeros((256, 256), dtype=np.uint8)
        gray[50:150, 50:150] = 200

        segmenter = LeafSegmenter()
        result = segmenter.run_otsu(gray)
        assert result is not None
        mask, area = result
        assert area > 0

    def test_leaf_segmenter_refine_with_contour(self):
        """Deve refinar máscara com contorno (F2.4)."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(mask, (128, 128), 50, 255, -1)

        segmenter = LeafSegmenter()
        result = segmenter.refine_with_contour(mask)
        assert result is not None
        refined_mask, area = result
        assert area > 0

    def test_leaf_segmenter_extract_leaf_region(self):
        """Deve extrair região foliar (F2.5)."""
        image = np.full((256, 256, 3), 100, dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[50:150, 50:150] = 255

        segmenter = LeafSegmenter()
        result = segmenter.extract_leaf_region(image, mask)
        assert result is not None
        assert result.shape == image.shape


if __name__ == "__main__":
    import cv2
    pytest.main([__file__, "-v"])

