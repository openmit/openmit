#!/usr/bin/env python
#-*- coding: utf-8 -*-

import exp_pb2 as exp

person = exp.Person()
person.name = "zhouyong";
person.id = 29;
person.xxxx = "xxxxx-yyyyy";
person.strs.append('strs0');
person.strs.append('strs1');

print('person:\n{}'.format(person))
