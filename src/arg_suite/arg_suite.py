# flake8: noqa
# Do NOT modify. To be replaced by an eternal dependency
"""Command-Line Arguments Parsing Suite

This module is an command-line argument library that:

    - generates argparse.ArgumentParser based a NamedTuple classed argument
    - handles two-way conversions: a typed argument object (A NamedTuple) <-> a command-line argv
    - enables IDE type hints and code auto-completion
    - promotes type-safety of command-line arguments


The following is a simple usage example that sums integers from the
command-line and writes the result to a file::


    @arg_suite
    class MyArg(NamedTuple):
        '''MyArg docstring goes to description
        Each named (without "_" prefix) and fully typed attribute will be turned into an ArgumentParser.add_argument
        '''

        nn: List[int]  # Comments go to ArgumentParser argument help
        _nn = {'choices': (200, 300)}  # (Advanced feature) Optional user supplement/override for ArgumentParser.add_argument("--nn", **_nn)

        a_tuple: Tuple[str, int]  # Arguments without defaults are treated as "required = True"
        h_param: Dict[str, int]  # Also supports List, Set

        ###### arguments without defaults go first as required by NamedTuple ######

        l_r: float = 0 # Arguments with defaults are treated as "required = False"
        n: Optional[int] = None


The module contains the following public classes:

- arg_suite -- The main entry point for command-line argument suite. As the
    example above shows, this decorator will attach an ArgSuite instance to
    the argument NamedTuple "subclass".

- ArgSuite -- The main class that generates the corresponding ArgumentParser
    and handles the two-way conversions.

All other classes and methods in this module are considered implementation details.
"""

__all__ = ('arg_suite', 'ArgSuite')


import inspect
import logging
import re
import tokenize
from argparse import *
from typing import *

PRIMITIVES = {str, int, float, bool}
SUPPORTED_OPTIONAL_TYPES = {Optional[p] for p in PRIMITIVES}
SUPPORTED_COLLECTION_TYPES = {Box[p] for Box in [List, Set] for p in PRIMITIVES} | {Dict[k, v] for k in
                                                                                    PRIMITIVES for v in
                                                                                    PRIMITIVES}
SUPPORTED_BOXED_TYPES = SUPPORTED_OPTIONAL_TYPES | SUPPORTED_COLLECTION_TYPES
SUPPORTED_TYPES = PRIMITIVES | SUPPORTED_OPTIONAL_TYPES | SUPPORTED_COLLECTION_TYPES | {Tuple}
RESTORE_OPTIONAL = re.compile(f'Union\\[({"|".join((p.__name__ for p in PRIMITIVES))}), NoneType\\]')
ArgType = TypeVar('ArgType', bound=Tuple)
logger = logging.getLogger(__name__)


def arg_suite(cls: Type[ArgType]) -> Type[ArgType]:
    """
    Decorator to get IDE type hint, hack the constructor, so that the IDE would infer the type of parsed_arg
    for code auto completion:
    > parsed_arg = ArgClass(my_argv)  # with this hack, no manual hint needed
    > parsed_arg: ArgClass = ArgClass.parse_to_arg(my_argv)  # need the manual type hint

    For NamedTuple can define but ignores untyped variables with defaults

    Must have types, required if no default value
    """
    suite = ArgSuite(cls)
    cls.arg_suite = suite
    __new = cls.__new__

    def __new__wrapper(arg_class, _argv: Optional[Sequence[str]] = None, **kwargs):
        # TODO Exception handling with helpful error message
        return __new(arg_class, **kwargs) if kwargs else suite.parse_to_arg(_argv)

    cls.__new__ = __new__wrapper
    cls.to_cmd_argv = lambda arg: suite.to_cmd_argv(arg)  # This needs manually add type hint
    cls.parse_to_arg = lambda argv=None: suite.parse_to_arg(argv)
    return cls


def is_arg_type(t):
    b = getattr(t, '__bases__', [])
    f = getattr(t, '_fields', [])
    f_t = getattr(t, '_field_types', {})
    return (len(b) == 1 and b[0] == tuple and isinstance(f, tuple) and isinstance(f_t, dict)
            and all(type(n) == str for n in f) and all(type(n) == str for n, _ in f_t.items()))


def get_origin(tp):
    return vars(tp).get('__origin__')


def get_args(tp):
    res = vars(tp).get('__args__')
    return res if res else ()


class ArgSuite(Generic[ArgType]):

    def __init__(self, arg_class: Type[ArgType]) -> None:
        assert is_arg_type(arg_class)
        self._arg_class = arg_class
        self._sub_parsers = {}
        self._parser = self._last_parser = ArgumentParser(description=self._arg_class.__doc__, fromfile_prefix_chars='@')
        self._parser.convert_arg_line_to_args = lambda arg_line: arg_line.split()
        self._gen_arguments_from_class(self._parser, arg_class)

    def _gen_arguments_from_class(self, parser: ArgumentParser, cls):
        field_defaults = cls._field_defaults
        fields = get_fields_meta(cls)
        for arg_name, arg_type in cls._field_types.items():
            kwargs = Namespace()
            # Build help message
            help_builder = ['(', type_to_str(arg_type)]
            # Get default if specified and set required if no default
            default = field_defaults.get(arg_name, ArgType)
            required = default == ArgType  # No default
            if required:
                kwargs.required = True
                help_builder.append(', required')
            elif True:  # TODO handle positional if we decided to support it later
                kwargs.default = default
                help_builder.append(', default: ')
                help_builder.append(str(default))
            help_builder.append(')')
            # Add from source code comment
            if arg_name in fields:
                help_builder.append(' ')
                help_builder.append(fields[arg_name].comment)
            kwargs.help = ''.join(help_builder)

            if is_arg_type(arg_type):
                raise NotImplementedError("Nested NamedTuple/arg not fully supported yet.")
                # TODO full support on the argv/cmdln generating side.
                # beware that sub_parsers options can override previously defined options siliently.
                # Need to decide which way to go with the implementation.
                assert arg_name not in self._sub_parsers
                self._last_parser = self._last_parser.add_subparsers(description=arg_type.__doc__, metavar=arg_name).add_parser(arg_name)
                self._sub_parsers[arg_name] = self._last_parser
                self._gen_arguments_from_class(self._last_parser, arg_type)
                continue

            # Use type for metavar in help display
            kwargs.metavar = type_to_str(arg_type)
            if get_origin(arg_type) in (Tuple, tuple):
                types = get_args(arg_type)
                if not types:
                    raise ArgumentTypeError(f'Invalid Tuple type: {arg_type} for "{arg_name}"')
                kwargs.nargs = len(types)
                kwargs.metavar = tuple(type_to_str(t) for t in types)
                type_gen = iter(types)
                arg_type = lambda item: next(type_gen)(item)
            elif arg_type in SUPPORTED_BOXED_TYPES:
                arg_types = get_args(arg_type)
                kwargs.nargs = '*'  # Consider using comment to specify the number of size of the collection
                if get_origin(arg_type) == dict:
                    if len(arg_types) == 2:
                        def arg_type(s: str, types=arg_types):
                            k, v = s.split(":")
                            return types[0](k), types[1](v)

                        kwargs.metavar = f'{type_to_str(arg_types[0])}:{type_to_str(arg_types[1])}'
                    else:
                        raise ArgumentTypeError(f'Invalid Dict type: {arg_type} for "{arg_name}"')
                elif arg_type in SUPPORTED_COLLECTION_TYPES:
                    arg_type = arg_types[0]
                    kwargs.metavar = type_to_str(arg_type)
                elif arg_type in SUPPORTED_OPTIONAL_TYPES:
                    del kwargs.nargs
                    arg_type = lambda s: None if s == 'None' else arg_types[0](s)
                    kwargs.metavar = type_to_str(arg_types[0])
                else:
                    raise ArgumentTypeError(f'Invalid Boxed type: {arg_type} for "{arg_name}"')
            elif arg_type not in SUPPORTED_TYPES:
                raise ArgumentTypeError(f'Unsupported type: {arg_type} for "{arg_name}"')

            if arg_type == bool:
                store_bool = False  # TODO use comment to specify, require change to the cmd generator
                if store_bool:
                    kwargs.action = f'store_{"true" if kwargs.required or not kwargs.default else "false"}'
                else:
                    arg_type = lambda s: True if s == 'True' else False if s == 'False' else s
                    kwargs.choices = [True, False]
            kwargs.type = arg_type
            kwargs.__dict__.update(**vars(cls).get(f'_{arg_name}', {}))  # apply user override
            if hasattr(kwargs, 'choices'):
                del kwargs.metavar  # 'metavar' would override 'choices' which makes the help message less helpful
            parser.add_argument(f'--{arg_name}', **vars(kwargs))

    def parse_to_arg(self, argv: Optional[Sequence[str]] = None) -> ArgType:
        ns = self._parser.parse_args(argv)
        logger.info(f"{argv} is parsed to {ns}")
        kwargs = {
            attr: get_origin(self._arg_class._field_types[attr])(value)
            if isinstance(value, List) else value
            for attr, value in vars(ns).items()
        }
        return self._arg_class(**kwargs)

    def to_cmd_argv(self, args: ArgType) -> Sequence[str]:
        return list(self._gen_cmd_argv(args._asdict()))

    @staticmethod
    def _gen_cmd_argv(args):
        for key, value in args.items():
            # TODO replace '--' with parser parameters
            yield f'--{key}'
            yield from ArgSuite._arg_to_strings(value)

    @staticmethod
    def _arg_to_strings(arg: Any):
        """Generator to yeild string from an argument to be used as a command-line argument."""
        if isinstance(arg, dict):
            yield from (f'{k}:{v}' for k, v in arg.items())
        elif not isinstance(arg, str) and isinstance(arg, Iterable):
            yield from (str(a) for a in arg)
        else:
            yield str(arg)

    def _format_argv(self, action, val) -> List[str]:
        """
        not used for now TODO Consider using this to format cmd line
        :param action:
        :type action:
        :param val:
        :type val:
        :return:
        :rtype:
        """
        nargs = action.nargs
        if nargs is None or nargs == OPTIONAL:
            result = [f'{val}']
        elif isinstance(nargs, int) or nargs == ZERO_OR_MORE or nargs == ONE_OR_MORE:
            result = [f'{it}' for it in val]
        elif nargs == REMAINDER:
            result = '...'
        elif nargs == PARSER:
            result = '%s ...' % val
        elif nargs == SUPPRESS:
            result = ''
        else:
            try:
                formats = ['%s' for _ in range(nargs)]
            except ArgumentTypeError:
                raise ValueError("invalid nargs value") from None
            result = ' '.join(formats) % val(nargs)
        return result


def type_to_str(t: Union[type, Type]) -> str:
    return t.__name__ if type(t) == type else RESTORE_OPTIONAL.sub('Optional[\\1]', str(t).replace('typing.', ''))

FieldMeta = NamedTuple('FieldMeta', [('comment', str)])

def get_fields_meta(cls: type):
    class Source:
        lines, _ = inspect.getsourcelines(cls)
        lines = enumerate(lines)

        def __call__(self):
            self.index, line = next(Source.lines)
            return line

    def line_tokenizer():
        source = Source()
        source()  # skip the `class` line and set source.index
        last_index = source.index
        for token in tokenize.generate_tokens(source):
            if last_index != source.index:
                yield '\n'  # New line indicator
            yield token
            last_index = source.index

    fields_meta = dict()
    field_column = field = last_token = None
    for token in line_tokenizer():
        if token == '\n':
            field = None  # New line
        elif token.type == tokenize.NAME:
            if field_column is None:
                field_column = token.start[1]
        elif token.exact_type == tokenize.COLON:
            if last_token.start[1] == field_column:
                # All fields are required to have type annotation so last_token is not None
                field = last_token.string
        elif token.type == tokenize.COMMENT and field:
            # TODO nicer way to deal with with long comments or support multiple lines
            fields_meta[field] = FieldMeta(comment=(token.string+' ')[1:token.string.lower().find('# noqa:')])
        last_token = token
    return fields_meta

def _raise(e: BaseException):
    raise e
