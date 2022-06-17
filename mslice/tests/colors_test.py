import unittest

from mslice.models.colors import color_to_name, name_to_color, named_cycle_colors, pretty_name


class ColorsTest(unittest.TestCase):

    def test_colors_list_is_limited_in_size(self):
        # checking that the list is not all of matplotlibs colors!
        self.assertLess(len(named_cycle_colors()), 15)

    def test_color_names_do_not_contain_prefixes(self):
        for name in named_cycle_colors():
            self.assertTrue(':' not in name)

    def test_known_color_name_gives_expected_hex(self):
        self.assertEqual("#2ca02c", name_to_color("green"))

    def test_known_hex_gives_expected_color_name(self):
        self.assertEqual("green", color_to_name("#2ca02c"))

    def test_unknown_color_name_raises_valueerror(self):
        self.assertRaises(ValueError, name_to_color, "NotAColorName")

    def test_unknown_hex_color_raises_valueerror(self):
        self.assertRaises(ValueError, color_to_name, "#12345")

    def test_basic_color_is_known(self):
        self.assertEqual('cyan', color_to_name('#17becf'))
        self.assertEqual('yellow', color_to_name('#ffff00'))

    def test_pretty_name(self):
        self.assertEqual('blue', pretty_name('tab:blue'))


if __name__ == '__main__':
    unittest.main()
