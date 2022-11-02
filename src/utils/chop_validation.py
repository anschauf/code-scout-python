import re


VALID_CHOP_REGEX = re.compile(r"^\d[A-Z0-9]{1,5}$")


def validate_chop_codes_list(chop_codes: list[str]) -> list[str]:
    """Check whether a list of CHOPs contains valid codes.

    @param chop_codes: The list of codes.
    @return: A list of valid codes only.
    """
    chop_codes_split = split_chop_codes(chop_codes)

    valid_chop_codes = list()

    for chop_info in chop_codes_split:
        chop = chop_info[0]

        # Remove the optional dots
        chop_without_dots = chop.replace('.', '').upper()

        matches = list(re.finditer(VALID_CHOP_REGEX, chop_without_dots))

        assert(len(matches) <= 1)
        if len(matches) == 1:
            valid_chop_code = matches[0].group()

            if len(chop_info) > 1:
                chop_info_to_concatenate = [valid_chop_code] + chop_info[1:]
            else:
                chop_info_to_concatenate = [valid_chop_code]

            valid_chop_code_info = ':'.join(chop_info_to_concatenate)
            valid_chop_codes.append(valid_chop_code_info)

    return valid_chop_codes


def split_chop_codes(codes_list: list[str]) -> list[list[str]]:
    """From a list of CHOP codes, which are formatted as '<code>:<side>:<date>', split them into their components.

    @param codes_list: The list of CHOP codes.
    @return: A list of the info for each code, split into strings.
    """
    return [code_with_colons.split(':') for code_with_colons in codes_list]
