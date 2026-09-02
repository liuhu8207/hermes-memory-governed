# -*- coding: utf-8 -*-
"""密钥闸门鲁棒性回归测试。

针对真实数据（state.db 989 条消息 + wiki 文件）发现的问题：
``key_value_secret`` pattern 的值部分过宽（``[^\\s,;，；]+``），把代码表达式
（``token = os.environ.get(...)``）、中文描述（``api_key: 你的key``）、
短值（``password=123456``）误判为密钥。

收紧后：值必须是"像密钥"的裸 token（字母数字开头、>=16 位，只含
``[A-Za-z0-9_\\-.]``）。这两个方向都要守住：误报 = 静默丢记忆，漏报 = 泄漏。
"""

from plugin.memory_governed._bridge import has_secret_like_text


class TestNoFalsePositive:
    """误报必须为 0（fail-closed 下误报 = 静默丢记忆）。"""

    def test_code_expression_not_secret(self):
        assert not has_secret_like_text('token = os.environ.get("CLAWHUB_TOKEN")')
        assert not has_secret_like_text("api_key = config.get('key')")
        assert not has_secret_like_text('token = "abc123"')
        # 属性访问（含点）不是密钥
        assert not has_secret_like_text("app_secret = settings.app_secret")
        assert not has_secret_like_text("self._app_secret = settings.app_secret")

    def test_chinese_description_not_secret(self):
        assert not has_secret_like_text("api_key: 你的key")
        assert not has_secret_like_text("token: 用于认证")

    def test_placeholder_not_secret(self):
        assert not has_secret_like_text("token=<your-token>")
        assert not has_secret_like_text("api_key: <填你的key>")

    def test_short_value_not_secret(self):
        assert not has_secret_like_text("password=123456")
        assert not has_secret_like_text("token=abc")


class TestNoFalseNegative:
    """真密钥仍要命中（漏报 = 泄漏）。"""

    def test_key_value_secret_still_detects(self):
        # 无 sk-/ghp_ 前缀，纯靠 key_value_secret 的值部分命中
        assert has_secret_like_text("api_key=AbCdEfGhIjKlMnOpQrSt")
        assert has_secret_like_text("FEISHU_APP_SECRET=ud7abcdefghijklmnopqrs")
        assert has_secret_like_text("token=AbCdEfGhIjKlMnOpQrStUvWxYz")

    def test_known_prefixes_still_detected(self):
        assert has_secret_like_text("sk-abcdefghijklmnopqrstuvw")
        assert has_secret_like_text("ghp_abcdefghijklmnopqrstuvw")
