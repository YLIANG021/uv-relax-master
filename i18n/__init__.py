import bpy

from . import (
    de_DE,
    en_US,
    es_ES,
    fr_FR,
    it_IT,
    ja_JP,
    ko_KR,
    pl_PL,
    pt_BR,
    ru_RU,
    vi_VN,
    zh_CN,
    zh_TW,
)


TRANSLATIONS = {
    "en": en_US.TRANSLATIONS,
    "en_US": en_US.TRANSLATIONS,
    "en_GB": en_US.TRANSLATIONS,
    "de": de_DE.TRANSLATIONS,
    "de_DE": de_DE.TRANSLATIONS,
    "es": es_ES.TRANSLATIONS,
    "es_ES": es_ES.TRANSLATIONS,
    "fr": fr_FR.TRANSLATIONS,
    "fr_FR": fr_FR.TRANSLATIONS,
    "it": it_IT.TRANSLATIONS,
    "it_IT": it_IT.TRANSLATIONS,
    "ja_JP": ja_JP.TRANSLATIONS,
    "ko_KR": ko_KR.TRANSLATIONS,
    "pl_PL": pl_PL.TRANSLATIONS,
    "pt": pt_BR.TRANSLATIONS,
    "pt_BR": pt_BR.TRANSLATIONS,
    "pt_PT": pt_BR.TRANSLATIONS,
    "ru_RU": ru_RU.TRANSLATIONS,
    "vi_VN": vi_VN.TRANSLATIONS,
    "zh_CN": zh_CN.TRANSLATIONS,
    "zh_HANS": zh_CN.TRANSLATIONS,
    "zh_TW": zh_TW.TRANSLATIONS,
    "zh_HANT": zh_TW.TRANSLATIONS,
}


def register():
    bpy.app.translations.register(__package__, TRANSLATIONS)


def unregister():
    try:
        bpy.app.translations.unregister(__package__)
    except RuntimeError:
        pass
