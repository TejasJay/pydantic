from __future__ import annotations

from dataclasses import dataclass
from pydantic_graph import BaseNode, GraphRunContext, End, Graph




@dataclass
class DivisibleBy5(BaseNode[None, None, int]):
    foo: int
    async def run(self, ctx: GraphRunContext) -> Increment | End[int]:
        if self.foo % 5 == 0:
            print("Divisible by 5: ",self.foo,'\n')
            return End(self.foo)
        else:
            print("Not Divisible so Incrementing: ",self.foo,'\n')
            return Increment(self.foo)
        

@dataclass
class Increment(BaseNode):
    foo: int
    async def run(self, ctx: GraphRunContext) -> DivisibleBy5:
        return DivisibleBy5(self.foo + 1)


fives_graph = Graph(nodes=[DivisibleBy5, Increment])
result = fives_graph.run_sync(DivisibleBy5(2))  
print("*"*25,"\nFINAL ANSWER: ",result.output,'\n')
